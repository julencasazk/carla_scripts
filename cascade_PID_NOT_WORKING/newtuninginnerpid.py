#!/usr/bin/env python3
"""
PSO tuning of the INNER acceleration controller (desired accel -> throttle/brake)
directly in CARLA using multiple vehicles in parallel.

What it does
  - Tunes a PID that maps desired longitudinal acceleration (m/s^2) to effort u.
  - Runs a short accel profile around a speed operating point and evaluates cost.

Parallel swarm execution
  - Each PSO particle is evaluated by a dedicated CARLA vehicle actor.
  - Only 8 spawn slots are used; if --particles > 8, evaluate in batches.

Cost terms (PSO objective)
  - Tracking error (IAE/SSE), overshoot, settling penalty
  - Control effort (u_rms) and rate (du_rms)
  - Saturation ratio, jerk penalty, throttle/brake switching penalty

Outputs
  - Prints PSO progress and best gains to stdout.
  - Optional per-particle logs/metrics to CSV (when enabled by flags).

Run (example)
  python3 cascade_PID_NOT_WORKING/newtuninginnerpid.py     --host localhost --port 2000 --particles 16 --iters 50 --seed 0
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import carla

from lib.PID import PID

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


# --------------------------- Helpers ---------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def fmt_vec(x: np.ndarray) -> str:
    return f"[kp={x[0]:.4f}, ki={x[1]:.4f}, kd={x[2]:.4f}, N={x[3]:.2f}]"


def loc_str(l: carla.Location) -> str:
    return f"({l.x:.1f},{l.y:.1f},{l.z:.1f})"


def get_v_long(vehicle: carla.Vehicle) -> float:
    v = vehicle.get_velocity()
    fwd = vehicle.get_transform().get_forward_vector()
    return float(v.x * fwd.x + v.y * fwd.y + v.z * fwd.z)


def lane_keep_steer(
    vehicle: carla.Vehicle,
    carla_map: carla.Map,
    lookahead_m: float = 12.0,
    steer_gain: float = 1.0,
    wheelbase_m: float = 2.8,
) -> float:
    tf = vehicle.get_transform()
    wp = carla_map.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return 0.0
    next_wps = wp.next(max(1.0, float(lookahead_m)))
    if not next_wps:
        return 0.0

    target = next_wps[0].transform.location
    vec = target - tf.location

    forward = tf.get_forward_vector()
    right = tf.get_right_vector()

    x = vec.x * forward.x + vec.y * forward.y + vec.z * forward.z
    y = vec.x * right.x + vec.y * right.y + vec.z * right.z

    if x <= 1e-3:
        return 0.0

    curvature = (2.0 * y) / (x * x + y * y)
    steer_angle = math.atan(wheelbase_m * curvature)
    return clamp(steer_gain * steer_angle, -1.0, 1.0)


# --------------------------- PSO ---------------------------

@dataclass
class Particle:
    x: np.ndarray      # [kp, ki, kd, N]
    v: np.ndarray
    pbest_x: np.ndarray
    pbest_J: float


def pso_init(bounds: List[Tuple[float, float]], n_particles: int, seed: int) -> List[Particle]:
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    particles: List[Particle] = []
    for _ in range(n_particles):
        x = np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=np.float64)
        v = np.zeros(dim, dtype=np.float64)
        particles.append(Particle(x=x, v=v, pbest_x=x.copy(), pbest_J=float("inf")))
    return particles


def pso_update(
    particles: List[Particle],
    gbest_x: np.ndarray,
    bounds: List[Tuple[float, float]],
    w: float,
    c1: float,
    c2: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    for p in particles:
        r1 = rng.random(dim)
        r2 = rng.random(dim)
        p.v = w * p.v + c1 * r1 * (p.pbest_x - p.x) + c2 * r2 * (gbest_x - p.x)
        p.x = p.x + p.v
        for i, (lo, hi) in enumerate(bounds):
            if p.x[i] < lo:
                p.x[i] = lo
                p.v[i] = 0.0
            elif p.x[i] > hi:
                p.x[i] = hi
                p.v[i] = 0.0


# --------------------------- Cost / Metrics ---------------------------

@dataclass
class TestLog:
    a_des: List[float]
    a_meas: List[float]
    u: List[float]
    v_long: List[float]


def compute_cost_from_log(
    log: TestLog,
    Ts: float,
    a_profile: List[Tuple[float, float]],
    w_track: float,
    w_os: float,
    w_settle: float,
    w_u: float,
    w_du: float,
    w_sat: float,
    w_jerk: float,
    w_switch: float,
) -> Tuple[float, Dict[str, float]]:
    a_des = np.asarray(log.a_des, dtype=np.float64)
    a = np.asarray(log.a_meas, dtype=np.float64)
    u = np.asarray(log.u, dtype=np.float64)

    if len(a) < 5:
        return 1e9, {"err": 1.0}

    # IMPORTANT:
    # We only score the nonzero pulse segments (abs(a_ref) > eps), not the baseline
    # segments where a_des == 0. In brake-only mode, the baseline segments are not
    # achievable without throttle due to drag, and including them biases gains.
    eps_ref = 1e-6

    iae = 0.0
    sse = 0.0
    u2_sum = 0.0
    du2_sum = 0.0
    sat_count = 0
    jerk2_sum = 0.0
    switch_count = 0
    n_u = 0
    n_du = 0
    n_jerk = 0
    scored_time_s = 0.0

    overshoot_sum = 0.0
    settle_pen = 0.0

    sign_eps = 0.02

    idx0 = 0
    n_total = len(a)
    for dur, a_ref in a_profile:
        n = int(round(dur / Ts))
        idx1 = min(n_total, idx0 + n)
        if idx1 <= idx0:
            break

        ref_val = float(a_ref)
        seg_a = a[idx0:idx1]
        seg_u = u[idx0:idx1]

        if abs(ref_val) > eps_ref:
            seg_e = (a_des[idx0:idx1] - seg_a)
            iae += float(np.sum(np.abs(seg_e)) * Ts)
            sse += float(np.sum(seg_e * seg_e) * Ts)

            u2_sum += float(np.sum(seg_u * seg_u))
            sat_count += int(np.sum(np.abs(seg_u) > 0.98))
            n_u += int(len(seg_u))
            scored_time_s += float(len(seg_u) * Ts)

            seg_du = np.diff(seg_u)
            if len(seg_du):
                du2_sum += float(np.sum(seg_du * seg_du))
                n_du += int(len(seg_du))

            seg_da = np.diff(seg_a)
            if len(seg_da):
                seg_jerk = seg_da / Ts
                jerk2_sum += float(np.sum(seg_jerk * seg_jerk))
                n_jerk += int(len(seg_jerk))

            sgn = np.sign(seg_u)
            sgn[np.abs(seg_u) < sign_eps] = 0.0
            for k in range(1, len(sgn)):
                if sgn[k] != 0 and sgn[k - 1] != 0 and sgn[k] != sgn[k - 1]:
                    switch_count += 1

            # Overshoot and settling are defined only for nonzero segments.
            if ref_val > 0:
                os = max(0.0, float(np.max(seg_a) - ref_val))
            else:
                os = max(0.0, float(ref_val - np.min(seg_a)))
            overshoot_sum += os / max(1e-6, abs(ref_val))

            band = 0.10 * abs(ref_val)
            within = np.abs(seg_a - ref_val) <= band
            settle_idx = None
            for j in range(len(within)):
                if within[j] and np.all(within[j:]):
                    settle_idx = j
                    break
            if settle_idx is None:
                settle_pen += 1.0
            else:
                settle_pen += (settle_idx * Ts) / max(1e-6, dur)

        idx0 = idx1

    if n_u <= 0:
        return 1e9, {"err": 1.0, "no_scored_samples": 1.0}

    u_rms = float(np.sqrt(u2_sum / max(1, n_u)))
    du_rms = float(np.sqrt(du2_sum / max(1, n_du))) if n_du > 0 else 0.0
    sat_ratio = float(sat_count / max(1, n_u))
    jerk_rms = float(np.sqrt(jerk2_sum / max(1, n_jerk))) if n_jerk > 0 else 0.0
    switch_rate = float(switch_count) / max(1e-6, scored_time_s)

    J = (
        w_track * (iae + 0.2 * sse)
        + w_os * overshoot_sum
        + w_settle * settle_pen
        + w_u * u_rms
        + w_du * du_rms
        + w_sat * sat_ratio
        + w_jerk * jerk_rms
        + w_switch * switch_rate
    )

    metrics = {
        "IAE": iae,
        "SSE": sse,
        "OS": overshoot_sum,
        "SETTLE": settle_pen,
        "u_rms": u_rms,
        "du_rms": du_rms,
        "sat": sat_ratio,
        "jerk_rms": jerk_rms,
        "switch_rate": switch_rate,
        "scored_time_s": float(scored_time_s),
    }
    return float(J), metrics


# --------------------------- CARLA Runner ---------------------------

@dataclass
class VehRuntime:
    actor: carla.Vehicle
    spawn: carla.Transform
    pid_inner: PID
    v_prev: float
    a_filt: float
    last_loc: carla.Location
    dist_m: float
    collided: bool


class CarlaPSOInnerPID:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.Ts = 0.01

        self.spawn_indices = [115, 116, 117, 118, 79, 80, 81, 82]
        self.n_slots = len(self.spawn_indices)

        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(30.0)

        print("[CARLA] Loading world Town04 ...")
        self.client.load_world("Town04")
        time.sleep(2.0)

        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()

        self.original_settings = self.world.get_settings()
        self._apply_settings()

        self.vehicles: List[VehRuntime] = []
        self.collision_sensors: List[carla.Sensor] = []
        self.camera: Optional[carla.Sensor] = None
        self.camera_image = None

        self._spawn_vehicles()
        self._print_spawn_debug()

    def _apply_settings(self) -> None:
        s = self.world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = self.Ts
        s.substepping = True
        s.max_substep_delta_time = 0.001
        s.max_substeps = 10
        s.no_rendering_mode = (not self.args.render)
        self.world.apply_settings(s)
        print(f"[CARLA] sync ON dt={self.Ts}, substep=0.001, no_render={s.no_rendering_mode}")

    def _print_spawn_debug(self) -> None:
        spawns = self.carla_map.get_spawn_points()
        print("[CARLA] Spawnpoints used:")
        for i, sp_idx in enumerate(self.spawn_indices):
            sp = spawns[sp_idx]
            print(f"  slot{i} -> spawn[{sp_idx}] loc={loc_str(sp.location)} yaw={sp.rotation.yaw:.1f}")

    def destroy(self) -> None:
        print("[CARLA] Destroying actors...")
        try:
            if self.camera is not None:
                self.camera.stop()
                self.camera.destroy()
        except Exception:
            pass
        for s in self.collision_sensors:
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass
        for vr in self.vehicles:
            try:
                vr.actor.destroy()
            except Exception:
                pass
        try:
            self.world.apply_settings(self.original_settings)
        except Exception:
            pass

    def _spawn_vehicles(self) -> None:
        spawns = self.carla_map.get_spawn_points()
        veh_bp = self.bp_lib.find(self.args.vehicle_bp)

        print("[CARLA] Spawning 8 vehicles (fixed slots)...")
        for slot_i, sp_idx in enumerate(self.spawn_indices):
            sp = spawns[sp_idx]
            sp.location.z += 0.35

            actor = self.world.try_spawn_actor(veh_bp, sp)
            if actor is None:
                raise RuntimeError(f"Failed to spawn vehicle at spawn index {sp_idx}")
            actor.set_autopilot(False)

            col_bp = self.bp_lib.find("sensor.other.collision")
            col = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=actor)

            pid_placeholder = PID(
                0.1,
                0.0,
                0.0,
                10.0,
                self.Ts,
                -1.0,
                1.0,
                1.0,
                der_on_meas=True,
                prop_on_meas=False,
                derivative_disc_method="tustin",
                integral_disc_method="tustin",
            )

            vr = VehRuntime(
                actor=actor,
                spawn=sp,
                pid_inner=pid_placeholder,
                v_prev=0.0,
                a_filt=0.0,
                last_loc=actor.get_transform().location,
                dist_m=0.0,
                collided=False,
            )

            def _make_on_col(vr_ref: VehRuntime, slot_idx: int):
                def _on_collision(event):
                    vr_ref.collided = True
                    if self.args.live_log:
                        other = getattr(event.other_actor, "type_id", "unknown")
                        print(f"[EVENT][COLLISION] slot{slot_idx} with {other}")
                return _on_collision

            col.listen(_make_on_col(vr, slot_i))

            self.vehicles.append(vr)
            self.collision_sensors.append(col)

        if self.args.render:
            if cv2 is None:
                print("[WARN] --render requested but cv2 not available. Continuing without dashcam.")
            else:
                idx = int(self.args.dashcam_vehicle)
                idx = max(0, min(idx, len(self.vehicles) - 1))
                self._attach_camera(self.vehicles[idx].actor)
                print(f"[CAM] Dashcam attached to slot {idx}")

        for _ in range(5):
            self.world.tick()

    def _attach_camera(self, vehicle: carla.Vehicle) -> None:
        cam_bp = self.bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.args.cam_w))
        cam_bp.set_attribute("image_size_y", str(self.args.cam_h))
        cam_bp.set_attribute("fov", str(self.args.cam_fov))
        tf = carla.Transform(carla.Location(x=-6.5, z=2.5), carla.Rotation(pitch=-10.0))
        self.camera = self.world.spawn_actor(cam_bp, tf, attach_to=vehicle)

        def _on_img(img: carla.Image):
            arr = np.frombuffer(img.raw_data, dtype=np.uint8)
            arr = arr.reshape((img.height, img.width, 4))[:, :, :3]
            self.camera_image = arr

        self.camera.listen(_on_img)

    def _render_dashcam(self) -> None:
        if self.camera_image is None or cv2 is None:
            return
        cv2.imshow("dashcam", self.camera_image)
        cv2.waitKey(1)

    def _teleport_reset(self, why: str = "") -> None:
        if why:
            print(f"[RESET] Teleporting all slots. Reason: {why}")
        for slot_i, vr in enumerate(self.vehicles):
            vr.actor.set_transform(vr.spawn)
            vr.actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            vr.actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            vr.pid_inner.reset()
            vr.v_prev = 0.0
            vr.a_filt = 0.0
            vr.last_loc = vr.actor.get_transform().location
            vr.dist_m = 0.0
            vr.collided = False
            vr.actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
            if self.args.live_log:
                tf = vr.actor.get_transform()
                print(f"  slot{slot_i} -> loc={loc_str(tf.location)} yaw={tf.rotation.yaw:.1f}")

        for _ in range(5):
            self.world.tick()
            if self.args.render:
                self._render_dashcam()

    def _update_odom(self, vr: VehRuntime) -> None:
        loc = vr.actor.get_transform().location
        dx = loc.x - vr.last_loc.x
        dy = loc.y - vr.last_loc.y
        dz = loc.z - vr.last_loc.z
        vr.dist_m += math.sqrt(dx * dx + dy * dy + dz * dz)
        vr.last_loc = loc

    def _compute_accel_filtered(self, vr: VehRuntime) -> Tuple[float, float, float]:
        v_long = get_v_long(vr.actor)
        a_raw = (v_long - vr.v_prev) / self.Ts
        vr.v_prev = v_long
        alpha = self.args.accel_lpf_alpha
        vr.a_filt = alpha * a_raw + (1.0 - alpha) * vr.a_filt
        return v_long, a_raw, vr.a_filt

    def _apply_lane_keep(self, vr: VehRuntime) -> float:
        return lane_keep_steer(
            vr.actor,
            self.carla_map,
            lookahead_m=self.args.lookahead_m,
            steer_gain=self.args.steer_gain,
            wheelbase_m=self.args.wheelbase_m,
        )

    # -------- Warm-up with OLD PID (gain scheduling) + gating for ACTIVE slots --------

    def _warmup_to_speed_old_pid(self, active_slots: List[int], v_op: float, max_time_s: float) -> bool:
        Ts = self.Ts
        u_min, u_max = -1.0, 1.0
        kb_aw = 1.0

        print(
            f"[WARMUP] active={len(active_slots)} v_op={v_op:.2f} m/s | max_time={max_time_s:.1f}s | "
            f"band=+/-{self.args.warm_band:.2f} | hold={self.args.warm_hold_s:.2f}s"
        )

        # Old gains (requested)
        slow_pid = PID(0.1089194,  0.04906409, 0.0,  7.71822214, Ts, u_min, u_max, kb_aw, der_on_meas=True, prop_on_meas=False, derivative_disc_method="tustin", integral_disc_method="tustin")
        mid_pid  = PID(0.11494122, 0.0489494,  0.0,  5.0,        Ts, u_min, u_max, kb_aw, der_on_meas=True, prop_on_meas=False, derivative_disc_method="tustin", integral_disc_method="tustin")
        fast_pid = PID(0.19021246, 0.15704399, 0.0, 12.70886655, Ts, u_min, u_max, kb_aw, der_on_meas=True, prop_on_meas=False, derivative_disc_method="tustin", integral_disc_method="tustin")

        v1 = float(self.args.warm_v1)
        v2 = float(self.args.warm_v2)
        h  = float(self.args.warm_hyst)

        # range state only for active slots
        ranges: Dict[int, str] = {s: "low" for s in active_slots}

        def select_pid(range_name: str) -> PID:
            if range_name == "high":
                return fast_pid
            if range_name == "mid":
                return mid_pid
            return slow_pid

        def update_range(cur: str, v: float) -> str:
            if cur == "low":
                return "mid" if v > v1 + h else "low"
            if cur == "mid":
                if v > v2 + h:
                    return "high"
                if v < v1 - h:
                    return "low"
                return "mid"
            return "mid" if v < v2 - h else "high"

        slow_pid.reset()
        mid_pid.reset()
        fast_pid.reset()

        hold_ticks_req = int(round(self.args.warm_hold_s / Ts))
        hold_ticks = 0
        band = float(self.args.warm_band)

        n_steps = int(max_time_s / Ts)

        # keep inactive vehicles braked
        inactive = [i for i in range(self.n_slots) if i not in active_slots]

        for k in range(n_steps):
            all_in_band = True

            if self.args.live_log and (k % self.args.warmup_log_every == 0):
                print(f"[WARMUP][t={k*Ts:6.2f}s] hold={hold_ticks}/{hold_ticks_req}")

            # park inactive
            for s in inactive:
                vr = self.vehicles[s]
                steer = self._apply_lane_keep(vr)
                vr.actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=steer))

            for s in active_slots:
                vr = self.vehicles[s]
                self._update_odom(vr)
                v_long = get_v_long(vr.actor)

                new_range = update_range(ranges[s], v_long)
                if new_range != ranges[s]:
                    ranges[s] = new_range
                    select_pid(new_range).reset()
                    if self.args.live_log:
                        print(f"  [SCHED] slot{s} -> {new_range} (v={v_long:.2f})")

                pid = select_pid(ranges[s])
                u = pid.step(v_op, v_long)

                steer = self._apply_lane_keep(vr)
                vr.actor.apply_control(carla.VehicleControl(
                    throttle=max(0.0, u),
                    brake=max(0.0, -u),
                    steer=steer,
                ))

                in_band = abs(v_long - v_op) <= band
                all_in_band = all_in_band and in_band

                if self.args.live_log and (k % self.args.warmup_log_every == 0):
                    tf = vr.actor.get_transform()
                    print(
                        f"  slot{s}: v={v_long:5.2f} range={ranges[s]:>4} u={u:+6.3f} "
                        f"thr={max(0,u):5.3f} brk={max(0,-u):5.3f} steer={steer:+6.3f} "
                        f"in_band={int(in_band)} dist={vr.dist_m:6.1f} loc={loc_str(tf.location)}"
                    )

                if vr.dist_m > self.args.max_dist_m:
                    if self.args.live_log:
                        print(f"[EVENT][DIST] slot{s} exceeded {self.args.max_dist_m}m during warmup (dist={vr.dist_m:.1f})")
                    return False

            self.world.tick()
            if self.args.render:
                self._render_dashcam()

            hold_ticks = hold_ticks + 1 if all_in_band else 0
            if hold_ticks >= hold_ticks_req:
                print(f"[WARMUP] OK: all ACTIVE slots in band for {self.args.warm_hold_s:.2f}s.")
                return True

        print("[WARMUP] FAILED (timeout).")
        return False

    # -------- Evaluate one batch (<=8 particles) --------

    def evaluate_batch(
        self,
        particles: List[Particle],
        particle_indices: List[int],
        a_profile: List[Tuple[float, float]],
        v_op: float,
        it_idx: int,
        batch_idx: int,
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        """
        Evaluate a subset of particles using the fixed 8 vehicle slots.
        particle_indices: indices in the global particles list to evaluate in this batch.
        Returns costs/metrics aligned to particle_indices order.
        """
        active_n = len(particle_indices)
        active_slots = list(range(active_n))  # map particle i -> slot i for this batch

        # Build timeline
        total_dur = sum(d for d, _ in a_profile)
        n_total = int(round(total_dur / self.Ts))
        a_timeline = np.zeros(n_total, dtype=np.float64)
        idx0 = 0
        for dur, a_ref in a_profile:
            n = int(round(dur / self.Ts))
            a_timeline[idx0:idx0 + n] = float(a_ref)
            idx0 += n

        # Constrain inner PID output depending on tuning mode.
        if self.args.tune_mode == "brake":
            u_min, u_max = -1.0, 0.0
        elif self.args.tune_mode == "throttle":
            u_min, u_max = 0.0, 1.0
        else:
            u_min, u_max = -1.0, 1.0

        self._teleport_reset(why=f"iter {it_idx} batch {batch_idx} (n={active_n})")

        ok = self._warmup_to_speed_old_pid(active_slots=active_slots, v_op=float(v_op), max_time_s=self.args.warm_max_time_s)
        if not ok:
            return [1e9] * active_n, [{"warmup_fail": 1.0}] * active_n

        if self.args.live_log:
            print("[ASSIGN] INNER PID params for active slots:")
        for local_i, p_idx in enumerate(particle_indices):
            vr = self.vehicles[local_i]
            kp, ki, kd, N = map(float, particles[p_idx].x.tolist())
            vr.pid_inner = PID(
                kp,
                ki,
                kd,
                N,
                self.Ts,
                u_min,
                u_max,
                1.0,
                der_on_meas=True,
                prop_on_meas=False,
                derivative_disc_method="tustin",
                integral_disc_method="tustin",
            )
            vr.pid_inner.reset()
            vr.v_prev = get_v_long(vr.actor)
            vr.a_filt = 0.0
            vr.dist_m = 0.0
            vr.last_loc = vr.actor.get_transform().location
            vr.collided = False
            if self.args.live_log:
                print(f"  slot{local_i} <- P{p_idx} {fmt_vec(particles[p_idx].x)}")

        if self.args.live_log:
            print(f"[TEST] iter={it_idx} batch={batch_idx} total={total_dur:.2f}s ticks={n_total}")

        logs: List[TestLog] = [TestLog(a_des=[], a_meas=[], u=[], v_long=[]) for _ in range(active_n)]
        abort_penalty = [0.0] * active_n
        sat_counts = [0] * active_n
        switch_counts = [0] * active_n
        prev_sign = [0] * active_n

        inactive_slots = list(range(active_n, self.n_slots))

        for k in range(n_total):
            a_des = float(a_timeline[k])

            # park inactive slots (brake)
            for s in inactive_slots:
                vr = self.vehicles[s]
                steer = self._apply_lane_keep(vr)
                vr.actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=steer))

            if self.args.live_log and (k % self.args.test_log_every == 0):
                print(f"[TEST][t={k*self.Ts:6.2f}s] a_des={a_des:+.2f}")

            for local_i in range(active_n):
                vr = self.vehicles[local_i]
                self._update_odom(vr)

                if vr.dist_m > self.args.max_dist_m:
                    abort_penalty[local_i] += 1e6
                if vr.collided:
                    abort_penalty[local_i] += 1e6

                v_long, a_raw, a_meas = self._compute_accel_filtered(vr)

                u = vr.pid_inner.step(a_des, a_meas)
                if abs(u) < self.args.u_deadband:
                    u = 0.0
                # Safety clamp (in case PID implementation doesn't strictly enforce bounds).
                if self.args.tune_mode == "brake":
                    u = clamp(float(u), -1.0, 0.0)
                elif self.args.tune_mode == "throttle":
                    u = clamp(float(u), 0.0, 1.0)
                else:
                    u = clamp(float(u), -1.0, 1.0)

                if abs(u) > 0.98:
                    sat_counts[local_i] += 1

                sign = 0
                if u > self.args.switch_eps:
                    sign = 1
                elif u < -self.args.switch_eps:
                    sign = -1
                if sign != 0 and prev_sign[local_i] != 0 and sign != prev_sign[local_i]:
                    switch_counts[local_i] += 1
                if sign != 0:
                    prev_sign[local_i] = sign

                steer = self._apply_lane_keep(vr)
                vr.actor.apply_control(carla.VehicleControl(
                    throttle=max(0.0, u),
                    brake=max(0.0, -u),
                    steer=steer,
                ))

                logs[local_i].a_des.append(a_des)
                logs[local_i].a_meas.append(a_meas)
                logs[local_i].u.append(u)
                logs[local_i].v_long.append(v_long)

                if v_long < self.args.min_v_long:
                    abort_penalty[local_i] += 5e5

                if self.args.live_log and (k % self.args.test_log_every == 0):
                    tf = vr.actor.get_transform()
                    print(
                        f"  slot{local_i}: v={v_long:5.2f} a_raw={a_raw:7.2f} a={a_meas:7.2f} "
                        f"u={u:+6.3f} thr={max(0,u):5.3f} brk={max(0,-u):5.3f} "
                        f"dist={vr.dist_m:6.1f} loc={loc_str(tf.location)}"
                    )

            self.world.tick()
            if self.args.render:
                self._render_dashcam()

        # Cost per active slot
        costs: List[float] = []
        metrics_list: List[Dict[str, float]] = []

        for local_i in range(active_n):
            J, metrics = compute_cost_from_log(
                log=logs[local_i],
                Ts=self.Ts,
                a_profile=a_profile,
                w_track=self.args.w_track,
                w_os=self.args.w_os,
                w_settle=self.args.w_settle,
                w_u=self.args.w_u,
                w_du=self.args.w_du,
                w_sat=self.args.w_sat,
                w_jerk=self.args.w_jerk,
                w_switch=self.args.w_switch,
            )

            metrics["abort_penalty"] = float(abort_penalty[local_i])
            metrics["sat_live"] = float(sat_counts[local_i] / max(1, n_total))
            metrics["switch_rate_live"] = float(switch_counts[local_i] / max(1.0, (n_total * self.Ts)))

            J_total = float(J + abort_penalty[local_i])
            costs.append(J_total)
            metrics_list.append(metrics)

        return costs, metrics_list


# --------------------------- Main ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    # CARLA
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--vehicle-bp", dest="vehicle_bp", type=str, default="vehicle.tesla.model3")

    # PSO swarm size (requested batching behavior)
    parser.add_argument("--particles", type=int, default=8, help="Number of PSO particles (swarm width). If >8, runs in batches of 8 per iteration.")

    # Render / camera
    parser.add_argument("--render", action="store_true", help="Enable rendering and dashcam window")
    parser.add_argument("--dashcam-vehicle", type=int, default=0)
    parser.add_argument("--cam-w", type=int, default=960)
    parser.add_argument("--cam-h", type=int, default=540)
    parser.add_argument("--cam-fov", type=int, default=90)

    # Logging
    parser.add_argument("--live-log", action="store_true", help="Verbose live logging during warmup/test")
    parser.add_argument("--warmup-log-every", type=int, default=50, help="Ticks between warmup live logs")
    parser.add_argument("--test-log-every", type=int, default=25, help="Ticks between test live logs")
    parser.add_argument("--switch-eps", type=float, default=0.02, help="Threshold for sign-switch count in u")

    # Lane keep
    parser.add_argument("--lookahead-m", type=float, default=12.0)
    parser.add_argument("--steer-gain", type=float, default=1.0)
    parser.add_argument("--wheelbase-m", type=float, default=2.8)

    # Accel estimation
    parser.add_argument("--accel-lpf-alpha", type=float, default=0.35)
    parser.add_argument("--min-v-long", type=float, default=1.0)

    # Warm-up (old PID) + gating
    parser.add_argument("--v-op", type=float, default=10.0)
    parser.add_argument("--warm-max-time-s", type=float, default=15.0)
    parser.add_argument("--warm-band", type=float, default=1.5)
    parser.add_argument("--warm-hold-s", type=float, default=1.0)
    parser.add_argument("--warm-v1", type=float, default=11.11)
    parser.add_argument("--warm-v2", type=float, default=22.22)
    parser.add_argument("--warm-hyst", type=float, default=1.0)
    parser.add_argument("--warm-kb-aw", type=float, default=1.0)

    # Reset logic
    parser.add_argument("--max-dist-m", type=float, default=250.0)

    # Inner PID
    parser.add_argument("--kb-aw", type=float, default=0.15)
    parser.add_argument("--inner-der-on-meas", action="store_true")
    parser.add_argument("--u-deadband", type=float, default=0.0)

    # PSO params
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--pso-w", type=float, default=0.6)
    parser.add_argument("--pso-c1", type=float, default=1.6)
    parser.add_argument("--pso-c2", type=float, default=1.6)

    # Cost weights
    parser.add_argument("--w-track", type=float, default=1.0)
    parser.add_argument("--w-os", type=float, default=2.0)
    parser.add_argument("--w-settle", type=float, default=1.0)
    parser.add_argument("--w-u", type=float, default=0.2)
    parser.add_argument("--w-du", type=float, default=0.4)
    parser.add_argument("--w-sat", type=float, default=2.0)
    parser.add_argument("--w-jerk", type=float, default=0.2)
    parser.add_argument("--w-switch", type=float, default=0.5)

    # a_des profile
    parser.add_argument("--a-plus", type=float, default=1.5)
    parser.add_argument("--a-minus", type=float, default=2.0)
    parser.add_argument("--t-pre", type=float, default=0.5)
    parser.add_argument("--t-step-plus", type=float, default=2.0)
    parser.add_argument("--t-mid", type=float, default=0.7)
    parser.add_argument("--t-step-minus", type=float, default=1.7)
    parser.add_argument("--t-post", type=float, default=1.0)
    parser.add_argument(
        "--tune-mode",
        choices=["both", "brake", "throttle"],
        default="both",
        help=(
            "Which regime to excite/tune. "
            "'both' = 0 -> +a_plus -> 0 -> -a_minus -> 0 (default). "
            "'brake' = 0 -> -a_minus -> 0 and constrain u to [-1,0]. "
            "'throttle' = 0 -> +a_plus -> 0 and constrain u to [0,1]."
        ),
    )

    args = parser.parse_args()

    # Bounds requested by user
    bounds = [(0.0, 1.0), (0.0, 5.0), (0.00, 0.05), (5.0, 20.0)]

    n_particles = int(args.particles)
    if n_particles < 1:
        raise SystemExit("--particles must be >= 1")

    if args.tune_mode == "brake":
        a_profile = [
            (args.t_pre, 0.0),
            (args.t_step_minus, -abs(args.a_minus)),
            (args.t_post, 0.0),
        ]
    elif args.tune_mode == "throttle":
        a_profile = [
            (args.t_pre, 0.0),
            (args.t_step_plus, +abs(args.a_plus)),
            (args.t_post, 0.0),
        ]
    else:
        a_profile = [
            (args.t_pre, 0.0),
            (args.t_step_plus, +abs(args.a_plus)),
            (args.t_mid, 0.0),
            (args.t_step_minus, -abs(args.a_minus)),
            (args.t_post, 0.0),
        ]

    particles = pso_init(bounds=bounds, n_particles=n_particles, seed=args.seed)
    gbest_x = particles[0].x.copy()
    gbest_J = float("inf")

    print(f"[INIT] particles={n_particles} (batches per iter = ceil(p/8) = {int(math.ceil(n_particles/8))})")
    print("[INIT] Initial swarm:")
    for i, p in enumerate(particles):
        print(f"  P{i} x={fmt_vec(p.x)}")

    runner = CarlaPSOInnerPID(args)

    try:
        for it in range(args.iters):
            print("\n" + "=" * 110)
            print(f"[PSO] ITER {it+1}/{args.iters}")

            # Evaluate in batches of 8
            batch_size = 8
            batches = int(math.ceil(n_particles / batch_size))

            all_costs = [float("inf")] * n_particles
            all_metrics: List[Dict[str, float]] = [{} for _ in range(n_particles)]

            for b in range(batches):
                start = b * batch_size
                end = min(n_particles, (b + 1) * batch_size)
                idxs = list(range(start, end))

                print(f"[BATCH] {b+1}/{batches} evaluating particles {start}..{end-1} (n={len(idxs)})")
                costs, metrics_list = runner.evaluate_batch(
                    particles=particles,
                    particle_indices=idxs,
                    a_profile=a_profile,
                    v_op=args.v_op,
                    it_idx=it+1,
                    batch_idx=b+1,
                )

                for local_i, p_idx in enumerate(idxs):
                    all_costs[p_idx] = float(costs[local_i])
                    all_metrics[p_idx] = metrics_list[local_i]

                # quick batch summary
                best_in_batch = min((all_costs[p] for p in idxs))
                print(f"[BATCH] done. best J in batch = {best_in_batch:.3f}")

            # Update pbest/gbest using all evaluated particles
            for i, p in enumerate(particles):
                J = float(all_costs[i])
                if J < p.pbest_J:
                    p.pbest_J = J
                    p.pbest_x = p.x.copy()
                if J < gbest_J:
                    gbest_J = J
                    gbest_x = p.x.copy()

            # Iter summary (compact)
            print("[ITER SUMMARY]")
            order = np.argsort(np.array(all_costs))
            topk = min(10, n_particles)
            for rank in range(topk):
                i = int(order[rank])
                m = all_metrics[i] if i < len(all_metrics) else {}
                print(
                    f"  #{rank+1:02d} P{i:03d} J={all_costs[i]:12.3f} x={fmt_vec(particles[i].x)} | "
                    f"IAE={m.get('IAE',0):.3f} OS={m.get('OS',0):.3f} sat={m.get('sat',0):.3f} abort={m.get('abort_penalty',0):.0f}"
                )
            print(f"[GBEST] J={gbest_J:.6f} x={fmt_vec(gbest_x)}")

            # PSO update step (one update per iter after evaluating all particles)
            pso_update(
                particles=particles,
                gbest_x=gbest_x,
                bounds=bounds,
                w=args.pso_w,
                c1=args.pso_c1,
                c2=args.pso_c2,
                seed=args.seed + 1000 + it,
            )

        print("\n" + "=" * 110)
        print("[DONE]")
        print(f"Best cost: {gbest_J:.6f}")
        print(f"Best params: {fmt_vec(gbest_x)}")

    finally:
        runner.destroy()
        if args.render and cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
