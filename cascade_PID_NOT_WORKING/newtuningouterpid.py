#!/usr/bin/env python3
"""
PSO tuning of the OUTER speed PID in a cascaded controller
(speed reference -> desired accel -> throttle/brake) directly in CARLA.

What it does
  - Tunes the outer PID mapping speed error to desired acceleration.
  - Inner accel PID(s) are fixed (single or split throttle/brake modes).
  - Evaluates multiple speed-step scenarios and sums scenario costs.

Parallel swarm execution
  - Each PSO particle is evaluated on a dedicated CARLA vehicle actor.
  - Only 8 spawn slots are used; if --particles > 8, evaluate in batches.

Cost terms (PSO objective)
  - Tracking error, overshoot, settling penalty
  - Control effort/rate, saturation ratio, jerk, switching penalties

Outputs
  - Prints PSO progress and best gains to stdout.
  - Optional per-particle logs/metrics to CSV (when enabled by flags).

Run (example)
  python3 cascade_PID_NOT_WORKING/newtuningouterpid.py     --host localhost --port 2000 --particles 32 --iters 100
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

# --------------------------- hardcoded defaults ---------------------------
#
# The goal is to keep CLI usage simple: you should only need host/port + PSO
# settings most of the time. Tweak the defaults below in-code as needed.

# Outer PID search space (kp, ki, kd, N) for PSO.
OUTER_BOUNDS_DEFAULT = {
    "kp": (0.0, 3.0),
    "ki": (0.0, 0.4),
    "kd": (0.0, 0.15),
    "N": (5.0, 20.0),
}

# Inner PID scheduling thresholds (m/s), matching Platooning.py / following_ros_cam.py.
SCHED_V1_MPS = 11.11  # 40 km/h
SCHED_V2_MPS = 22.22  # 80 km/h
SCHED_HYST_MPS = 1.0

# Fixed inner PID gains per speed range.
# Format: (kp, ki, kd, N)
#
# If you use split-inner (recommended for separate throttle vs brake plants),
# fill THROTTLE and BRAKE tables. If you use single-inner, fill SINGLE table.
INNER_SINGLE_GAINS = {
    "low": (0.0238, 0.7760, 0.0019, 15.75),
    "mid": (0.0877, 0.6325, 0.0044, 20.0),
    "high": (0.0573, 0.4671, 0.00, 20.0),
}
INNER_THROTTLE_GAINS = {
    "low": (0.1898, 1.3466, 0.0028, 20.0),
    "mid": (0.2362, 1.3336, 0.0056, 20.0),
    "high": (0.2142, 1.5168, 0.0049, 20.0),
}
INNER_BRAKE_GAINS = {
    "low": (0.1869, 1.8861, 0.0002, 12.39),
    "mid": (0.0613, 1.9646, 0.0165, 13.44),
    "high": (0.2721, 4.4118, 0.049, 17.24),
}

# Default to split inner PIDs (throttle/brake) to match your plant identification.
SPLIT_INNER_DEFAULT = False


# --------------------------- helpers ---------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def fmt_vec(x: np.ndarray) -> str:
    return f"[kp={x[0]:.5f}, ki={x[1]:.5f}, kd={x[2]:.5f}, N={x[3]:.2f}]"


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

    nxt = wp.next(max(1.0, float(lookahead_m)))
    if not nxt:
        return 0.0

    target = nxt[0].transform.location
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

def update_speed_range(current: str, speed_abs: float, v1: float, v2: float, h: float) -> Tuple[str, bool]:
    prev = current
    if current == "low":
        if speed_abs > v1 + h:
            current = "mid"
    elif current == "mid":
        if speed_abs > v2 + h:
            current = "high"
        elif speed_abs < v1 - h:
            current = "low"
    else:  # high
        if speed_abs < v2 - h:
            current = "mid"
    return current, (current != prev)

def mode_from_effort(mode: str, u_effort: float, db_on: float, db_off: float) -> str:
    if mode == "coast":
        if u_effort > db_on:
            return "throttle"
        if u_effort < -db_on:
            return "brake"
        return "coast"
    if mode == "throttle":
        return "coast" if u_effort < db_off else "throttle"
    return "coast" if u_effort > -db_off else "brake"

def parse_pid4(spec: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if spec is None:
        return None
    parts = [p.strip() for p in str(spec).split(",") if p.strip() != ""]
    if len(parts) != 4:
        raise ValueError(f"Expected 'kp,ki,kd,N' but got: {spec!r}")
    kp, ki, kd, N = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    return kp, ki, kd, N


# --------------------------- PSO ---------------------------

@dataclass
class Particle:
    x: np.ndarray
    v: np.ndarray
    pbest_x: np.ndarray
    pbest_J: float


def pso_init(bounds: List[Tuple[float, float]], n_particles: int, seed: int) -> List[Particle]:
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    parts: List[Particle] = []
    for _ in range(n_particles):
        x = np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=np.float64)
        v = np.zeros(dim, dtype=np.float64)
        parts.append(Particle(x=x, v=v, pbest_x=x.copy(), pbest_J=float("inf")))
    return parts


def pso_update(
    particles: List[Particle],
    gbest_x: np.ndarray,
    bounds: List[Tuple[float, float]],
    w: float, c1: float, c2: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    for p in particles:
        r1 = rng.random(dim)
        r2 = rng.random(dim)
        p.v = w * p.v + c1 * r1 * (p.pbest_x - p.x) + c2 * r2 * (gbest_x - p.x)
        p.x = p.x + p.v

        # clamp to bounds
        for i, (lo, hi) in enumerate(bounds):
            if p.x[i] < lo:
                p.x[i] = lo
                p.v[i] = 0.0
            elif p.x[i] > hi:
                p.x[i] = hi
                p.v[i] = 0.0


# --------------------------- scenarios / cost ---------------------------

@dataclass
class ScenarioSpec:
    name: str
    v_op: float
    dv: float
    t_pre: float
    t_step: float
    t_post: float


def scenario_metrics(
    Ts: float,
    v_ref: np.ndarray,
    v: np.ndarray,
    a_des: np.ndarray,
    u: np.ndarray,
    dv: float,
    settle_pct: float,
    settle_abs_min: float,
) -> Dict[str, float]:
    e = v_ref - v
    iae = float(np.sum(np.abs(e)) * Ts)
    sse = float(np.sum(e * e) * Ts)

    a_rms = float(np.sqrt(np.mean(a_des * a_des)))
    da = np.diff(a_des)
    da_rms = float(np.sqrt(np.mean(da * da))) if len(da) else 0.0

    sat_u = float(np.mean(np.abs(u) > 0.98))

    eps = 0.02
    sgn = np.sign(u)
    sgn[np.abs(u) < eps] = 0.0
    switches = 0
    prev = 0.0
    for k in range(len(sgn)):
        if sgn[k] == 0:
            continue
        if prev != 0.0 and sgn[k] != prev:
            switches += 1
        prev = sgn[k]
    switch_rate = float(switches) / max(1.0, len(u) * Ts)

    # detect step segment boundaries from v_ref changes
    n = len(v_ref)
    k_step_start = 0
    for k in range(1, n):
        if abs(v_ref[k] - v_ref[k - 1]) > 1e-9:
            k_step_start = k
            break
    k_step_end = n
    for k in range(k_step_start + 1, n):
        if abs(v_ref[k] - v_ref[k - 1]) > 1e-9:
            k_step_end = k
            break

    v0 = float(v_ref[0])
    v1 = float(v0 + dv)
    seg = v[k_step_start:k_step_end]

    os_rel = 0.0
    settle_pen = 0.0
    if len(seg) >= 3 and abs(dv) > 1e-6:
        if dv > 0:
            os = max(0.0, float(np.max(seg) - v1))
        else:
            os = max(0.0, float(v1 - np.min(seg)))
        os_rel = os / max(1e-6, abs(dv))

        band = max(settle_abs_min, settle_pct * abs(dv))
        within = np.abs(seg - v1) <= band
        settle_idx = None
        for j in range(len(within)):
            if within[j] and np.all(within[j:]):
                settle_idx = j
                break
        if settle_idx is None:
            settle_pen = 1.0
        else:
            settle_pen = (settle_idx * Ts) / max(1e-6, (len(seg) * Ts))

    return {
        "IAE": iae,
        "SSE": sse,
        "OS": float(os_rel),
        "SETTLE": float(settle_pen),
        "a_rms": a_rms,
        "da_rms": da_rms,
        "sat_u": sat_u,
        "switch_rate": switch_rate,
    }


# --------------------------- CARLA runtime ---------------------------

@dataclass
class Slot:
    actor: carla.Vehicle
    spawn: carla.Transform

    # estimators / state
    v_prev: float
    a_filt: float
    last_loc: carla.Location
    dist_m: float

    # controllers (reused; reset between scenarios)
    outer: PID
    # Inner PID gain scheduling: low/mid/high (single or split)
    inner_low: PID
    inner_mid: PID
    inner_high: PID
    inner_thr_low: Optional[PID]
    inner_thr_mid: Optional[PID]
    inner_thr_high: Optional[PID]
    inner_brk_low: Optional[PID]
    inner_brk_mid: Optional[PID]
    inner_brk_high: Optional[PID]

    current_range: str
    mode: str


class CarlaOuterTuner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.Ts = 0.01
        # Scheduling thresholds (match Platooning.py / following_ros_cam.py)
        self.v1 = float(getattr(args, "sched_v1", SCHED_V1_MPS))
        self.v2 = float(getattr(args, "sched_v2", SCHED_V2_MPS))
        self.h = float(getattr(args, "sched_hyst", SCHED_HYST_MPS))
        self.spawn_indices = [115, 116, 117, 118, 79, 80, 81, 82]
        self.n_slots = len(self.spawn_indices)

        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(30.0)

        print("[CARLA] Loading Town04 ...")
        self.client.load_world("Town04")
        time.sleep(1.2)

        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()
        self.original_settings = self.world.get_settings()

        self._apply_settings()

        self.camera: Optional[carla.Sensor] = None
        self.camera_image = None

        self.slots: List[Slot] = []
        self._spawn_slots()
        self._attach_dashcam_if_needed()

        for _ in range(10):
            self.world.tick()
            self._render("READY")

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

    def destroy(self) -> None:
        print("[CARLA] Destroying actors...")
        try:
            if self.camera is not None:
                self.camera.stop()
                self.camera.destroy()
        except Exception:
            pass
        for s in self.slots:
            try:
                s.actor.destroy()
            except Exception:
                pass
        try:
            self.world.apply_settings(self.original_settings)
        except Exception:
            pass
        if self.args.render and cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _spawn_slots(self) -> None:
        spawns = self.carla_map.get_spawn_points()
        veh_bp = self.bp_lib.find(self.args.vehicle_bp)

        # Prepare inner PID gains per speed range.
        # If per-range strings are not provided, default to the base args.
        inner_low = parse_pid4(getattr(self.args, "inner_low", None))
        inner_mid = parse_pid4(getattr(self.args, "inner_mid", None))
        inner_high = parse_pid4(getattr(self.args, "inner_high", None))

        inner_thr_low = parse_pid4(getattr(self.args, "inner_thr_low", None))
        inner_thr_mid = parse_pid4(getattr(self.args, "inner_thr_mid", None))
        inner_thr_high = parse_pid4(getattr(self.args, "inner_thr_high", None))
        inner_brk_low = parse_pid4(getattr(self.args, "inner_brk_low", None))
        inner_brk_mid = parse_pid4(getattr(self.args, "inner_brk_mid", None))
        inner_brk_high = parse_pid4(getattr(self.args, "inner_brk_high", None))

        base_single = (float(self.args.inner_kp), float(self.args.inner_ki), float(self.args.inner_kd), float(self.args.inner_N))
        base_thr = (float(self.args.inner_thr_kp), float(self.args.inner_thr_ki), float(self.args.inner_thr_kd), float(self.args.inner_thr_N))
        base_brk = (float(self.args.inner_brk_kp), float(self.args.inner_brk_ki), float(self.args.inner_brk_kd), float(self.args.inner_brk_N))

        inner_low = inner_low or base_single
        inner_mid = inner_mid or base_single
        inner_high = inner_high or base_single
        inner_thr_low = inner_thr_low or base_thr
        inner_thr_mid = inner_thr_mid or base_thr
        inner_thr_high = inner_thr_high or base_thr
        inner_brk_low = inner_brk_low or base_brk
        inner_brk_mid = inner_brk_mid or base_brk
        inner_brk_high = inner_brk_high or base_brk

        print("[CARLA] Spawning 8 slot vehicles ...")
        for i, sp_idx in enumerate(self.spawn_indices):
            sp = spawns[sp_idx]
            sp.location.z += 0.35
            actor = self.world.try_spawn_actor(veh_bp, sp)
            if actor is None:
                raise RuntimeError(f"Failed to spawn at spawn index {sp_idx}")
            actor.set_autopilot(False)

            # placeholder controllers; outer gains overwritten per particle before running scenarios
            outer = PID(
                0.0,
                0.0,
                0.0,
                10.0,
                self.Ts,
                -float(self.args.a_max_brake),
                +float(self.args.a_max_throttle),
                1.0,
                der_on_meas=True,
                prop_on_meas=False,
                derivative_disc_method="tustin",
                integral_disc_method="tustin",
            )
            outer.reset()

            # Always build scheduled single-inner PIDs (used when split_inner is False).
            in_lo = PID(*inner_low, self.Ts, -1.0, 1.0, 1.0, der_on_meas=True,
                        prop_on_meas=False, derivative_disc_method="tustin", integral_disc_method="tustin")
            in_md = PID(*inner_mid, self.Ts, -1.0, 1.0, 1.0, der_on_meas=True,
                        prop_on_meas=False, derivative_disc_method="tustin", integral_disc_method="tustin")
            in_hi = PID(*inner_high, self.Ts, -1.0, 1.0, 1.0, der_on_meas=True,
                        prop_on_meas=False, derivative_disc_method="tustin", integral_disc_method="tustin")
            in_lo.reset()
            in_md.reset()
            in_hi.reset()

            # Optionally build scheduled split-inner PIDs.
            thr_lo = thr_md = thr_hi = None
            brk_lo = brk_md = brk_hi = None
            if bool(self.args.split_inner):
                thr_lo = PID(*inner_thr_low, self.Ts, 0.0, 1.0, 1.0,
                             der_on_meas=True,
                             prop_on_meas=False, derivative_disc_method="tustin", integral_disc_method="tustin")
                thr_md = PID(*inner_thr_mid, self.Ts, 0.0, 1.0, 1.0,
                             der_on_meas=True,
                             prop_on_meas=False, derivative_disc_method="tustin", integral_disc_method="tustin")
                thr_hi = PID(*inner_thr_high, self.Ts, 0.0, 1.0, 1.0,
                             der_on_meas=True,
                             prop_on_meas=False, derivative_disc_method="tustin", integral_disc_method="tustin")
                brk_lo = PID(*inner_brk_low, self.Ts, -1.0, 0.0, 1.0,
                             der_on_meas=True,
                             prop_on_meas=False, derivative_disc_method="tustin", integral_disc_method="tustin")
                brk_md = PID(*inner_brk_mid, self.Ts, -1.0, 0.0, 1.0,
                             der_on_meas=True,
                             prop_on_meas=False, derivative_disc_method="tustin", integral_disc_method="tustin")
                brk_hi = PID(*inner_brk_high, self.Ts, -1.0, 0.0, 1.0,
                             der_on_meas=True,
                             prop_on_meas=False, derivative_disc_method="tustin", integral_disc_method="tustin")
                thr_lo.reset()
                thr_md.reset()
                thr_hi.reset()
                brk_lo.reset()
                brk_md.reset()
                brk_hi.reset()

            self.slots.append(Slot(
                actor=actor,
                spawn=sp,
                v_prev=get_v_long(actor),
                a_filt=0.0,
                last_loc=actor.get_transform().location,
                dist_m=0.0,
                outer=outer,
                inner_low=in_lo,
                inner_mid=in_md,
                inner_high=in_hi,
                inner_thr_low=thr_lo,
                inner_thr_mid=thr_md,
                inner_thr_high=thr_hi,
                inner_brk_low=brk_lo,
                inner_brk_mid=brk_md,
                inner_brk_high=brk_hi,
                current_range="low",
                mode="coast",
            ))

            print(f"  slot{i} -> spawn[{sp_idx}] loc={loc_str(sp.location)} yaw={sp.rotation.yaw:.1f}")

    def _attach_dashcam_if_needed(self) -> None:
        if not self.args.render:
            return
        if cv2 is None:
            print("[WARN] --render requested but cv2 not available.")
            return

        idx = max(0, min(int(self.args.dashcam_vehicle), self.n_slots - 1))
        cam_bp = self.bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.args.cam_w))
        cam_bp.set_attribute("image_size_y", str(self.args.cam_h))
        cam_bp.set_attribute("fov", str(self.args.cam_fov))

        tf = carla.Transform(carla.Location(x=-6.5, z=2.5), carla.Rotation(pitch=-10.0))
        self.camera = self.world.spawn_actor(cam_bp, tf, attach_to=self.slots[idx].actor)

        def _on_img(img: carla.Image):
            arr = np.frombuffer(img.raw_data, dtype=np.uint8)
            self.camera_image = arr.reshape((img.height, img.width, 4))[:, :, :3]

        self.camera.listen(_on_img)
        print(f"[CAM] Dashcam attached to slot {idx}")

    def _render(self, overlay: str) -> None:
        if not self.args.render or cv2 is None or self.camera_image is None:
            return
        img = self.camera_image.copy()
        if overlay:
            cv2.putText(img, overlay, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("outer PID PSO", img)
        cv2.waitKey(1)

    # ----- reset / estimators -----

    def _reset_slots_soft(self, active_slots: List[int], why: str) -> None:
        # Soft reset: teleport + zero velocity + brake briefly; NO destroy.
        if self.args.verbose:
            print(f"[RESET] slots={active_slots} why={why}")

        for si in active_slots:
            s = self.slots[si]
            s.actor.set_transform(s.spawn)
            s.actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            s.actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            s.actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))

            s.v_prev = 0.0
            s.a_filt = 0.0
            s.last_loc = s.actor.get_transform().location
            s.dist_m = 0.0

            s.outer.reset()
            s.inner_low.reset()
            s.inner_mid.reset()
            s.inner_high.reset()
            if s.inner_thr_low is not None:
                s.inner_thr_low.reset()
            if s.inner_thr_mid is not None:
                s.inner_thr_mid.reset()
            if s.inner_thr_high is not None:
                s.inner_thr_high.reset()
            if s.inner_brk_low is not None:
                s.inner_brk_low.reset()
            if s.inner_brk_mid is not None:
                s.inner_brk_mid.reset()
            if s.inner_brk_high is not None:
                s.inner_brk_high.reset()
            s.current_range = "low"
            s.mode = "coast"

        # settle
        for _ in range(8):
            self.world.tick()
            self._render("RESET")

        # release brake
        for si in active_slots:
            s = self.slots[si]
            s.actor.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
        for _ in range(2):
            self.world.tick()
            self._render("RESET")

    def _park_inactive(self, inactive_slots: List[int]) -> None:
        for si in inactive_slots:
            s = self.slots[si]
            steer = lane_keep_steer(
                s.actor, self.carla_map,
                lookahead_m=self.args.lookahead_m,
                steer_gain=self.args.steer_gain,
                wheelbase_m=self.args.wheelbase_m,
            )
            s.actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=steer))

    def _update_dist(self, s: Slot) -> None:
        loc = s.actor.get_transform().location
        dx = loc.x - s.last_loc.x
        dy = loc.y - s.last_loc.y
        dz = loc.z - s.last_loc.z
        s.dist_m += math.sqrt(dx * dx + dy * dy + dz * dz)
        s.last_loc = loc

    def _accel_filtered(self, s: Slot) -> Tuple[float, float, float]:
        v_long = get_v_long(s.actor)
        a_raw = (v_long - s.v_prev) / self.Ts
        s.v_prev = v_long
        alpha = float(self.args.accel_lpf_alpha)
        s.a_filt = alpha * a_raw + (1.0 - alpha) * s.a_filt
        return v_long, a_raw, s.a_filt

    # ----- warmup (OLD speed PID scheduling) -----

    def _warmup_to_speed_old_pid(self, active_slots: List[int], v_op: float) -> bool:
        Ts = self.Ts
        u_min, u_max = -1.0, 1.0
        kb_aw = 1.0

        slow_pid = PID(0.1089194,  0.04906409, 0.0,  7.71822214, Ts, u_min, u_max, kb_aw, der_on_meas=True, prop_on_meas=False, derivative_disc_method="tustin", integral_disc_method="tustin")
        mid_pid  = PID(0.11494122, 0.0489494,  0.0,  5.0,        Ts, u_min, u_max, kb_aw, der_on_meas=True, prop_on_meas=False, derivative_disc_method="tustin", integral_disc_method="tustin")
        fast_pid = PID(0.19021246, 0.15704399, 0.0, 12.70886655, Ts, u_min, u_max, kb_aw, der_on_meas=True, prop_on_meas=False, derivative_disc_method="tustin", integral_disc_method="tustin")
        slow_pid.reset(); mid_pid.reset(); fast_pid.reset()

        v1 = float(self.args.warm_v1)
        v2 = float(self.args.warm_v2)
        h  = float(self.args.warm_hyst)

        ranges: Dict[int, str] = {si: "low" for si in active_slots}

        def select_pid(r: str) -> PID:
            return fast_pid if r == "high" else (mid_pid if r == "mid" else slow_pid)

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

        hold_ticks_req = int(round(float(self.args.warm_hold_s) / Ts))
        hold_ticks = 0
        band = float(self.args.warm_band)
        n_steps = int(round(float(self.args.warm_max_time_s) / Ts))

        for k in range(n_steps):
            all_in_band = True

            for si in active_slots:
                s = self.slots[si]
                self._update_dist(s)
                if s.dist_m > float(self.args.max_dist_m):
                    return False

                v_long = get_v_long(s.actor)
                new_range = update_range(ranges[si], v_long)
                if new_range != ranges[si]:
                    ranges[si] = new_range
                    select_pid(new_range).reset()

                pid = select_pid(ranges[si])
                u = float(pid.step(v_op, v_long))

                steer = lane_keep_steer(
                    s.actor, self.carla_map,
                    lookahead_m=self.args.lookahead_m,
                    steer_gain=self.args.steer_gain,
                    wheelbase_m=self.args.wheelbase_m,
                )
                s.actor.apply_control(carla.VehicleControl(
                    throttle=max(0.0, u),
                    brake=max(0.0, -u),
                    steer=steer,
                ))

                all_in_band = all_in_band and (abs(v_long - v_op) <= band)

            self.world.tick()
            self._render(f"WARMUP v_op={v_op:.1f}")

            hold_ticks = hold_ticks + 1 if all_in_band else 0
            if hold_ticks >= hold_ticks_req:
                return True

        return False

    # ----- batch evaluation -----

    def evaluate_batch(
        self,
        particles: List[Particle],
        particle_indices: List[int],
        scenarios: List[ScenarioSpec],
        it_idx: int,
        batch_idx: int,
    ) -> List[float]:
        active_n = len(particle_indices)
        active_slots = list(range(active_n))
        inactive_slots = list(range(active_n, self.n_slots))

        # Per-batch reset (soft)
        self._reset_slots_soft(active_slots, why=f"iter {it_idx} batch {batch_idx} start")

        # Assign outer gains for this batch; inner is already fixed in Slot
        for local_i, p_idx in enumerate(particle_indices):
            kp, ki, kd, N = map(float, particles[p_idx].x.tolist())
            s = self.slots[local_i]
            s.outer = PID(
                float(kp),
                float(ki),
                float(kd),
                float(N),
                self.Ts,
                -float(self.args.a_max_brake),
                +float(self.args.a_max_throttle),
                1.0,
                der_on_meas=True,
                prop_on_meas=False,
                derivative_disc_method="tustin",
                integral_disc_method="tustin",
            )
            s.outer.reset()
            s.inner_low.reset()
            s.inner_mid.reset()
            s.inner_high.reset()
            if s.inner_thr_low is not None:
                s.inner_thr_low.reset()
            if s.inner_thr_mid is not None:
                s.inner_thr_mid.reset()
            if s.inner_thr_high is not None:
                s.inner_thr_high.reset()
            if s.inner_brk_low is not None:
                s.inner_brk_low.reset()
            if s.inner_brk_mid is not None:
                s.inner_brk_mid.reset()
            if s.inner_brk_high is not None:
                s.inner_brk_high.reset()
            s.mode = "coast"

            s.v_prev = get_v_long(s.actor)
            s.a_filt = 0.0
            s.dist_m = 0.0
            s.last_loc = s.actor.get_transform().location

        total_cost = [0.0] * active_n
        debug_breakdown: Optional[List[List[Dict[str, float]]]] = None
        if bool(getattr(self.args, "debug_metrics", False)):
            debug_breakdown = [[] for _ in range(active_n)]

        for sc in scenarios:
            if self.args.verbose:
                print(f"[SCENARIO] {sc.name} v_op={sc.v_op:.1f} dv={sc.dv:+.1f}")

            # scenario reset + warmup
            self._reset_slots_soft(active_slots, why=f"{sc.name} start")
            self._park_inactive(inactive_slots)

            ok = self._warmup_to_speed_old_pid(active_slots, sc.v_op)
            if not ok:
                for i in range(active_n):
                    total_cost[i] += 1e7
                continue

            # reset controller state at scenario start
            for si in active_slots:
                s = self.slots[si]
                s.outer.reset()
                s.inner_low.reset()
                s.inner_mid.reset()
                s.inner_high.reset()
                if s.inner_thr_low is not None:
                    s.inner_thr_low.reset()
                if s.inner_thr_mid is not None:
                    s.inner_thr_mid.reset()
                if s.inner_thr_high is not None:
                    s.inner_thr_high.reset()
                if s.inner_brk_low is not None:
                    s.inner_brk_low.reset()
                if s.inner_brk_mid is not None:
                    s.inner_brk_mid.reset()
                if s.inner_brk_high is not None:
                    s.inner_brk_high.reset()
                s.v_prev = get_v_long(s.actor)
                s.a_filt = 0.0
                s.dist_m = 0.0
                s.last_loc = s.actor.get_transform().location
                s.current_range = "low"
                s.mode = "coast"

            # build v_ref timeline
            def rep(dur: float, val: float) -> List[float]:
                return [val] * int(round(dur / self.Ts))

            v0 = float(sc.v_op)
            v1 = float(sc.v_op + sc.dv)
            v_ref_list = rep(sc.t_pre, v0) + rep(sc.t_step, v1) + rep(sc.t_post, v0)
            n_total = len(v_ref_list)

            # buffers per active slot
            Vref = [np.zeros(n_total, dtype=np.float64) for _ in range(active_n)]
            V = [np.zeros(n_total, dtype=np.float64) for _ in range(active_n)]
            Ades = [np.zeros(n_total, dtype=np.float64) for _ in range(active_n)]
            U = [np.zeros(n_total, dtype=np.float64) for _ in range(active_n)]

            # jerk limit for a_des
            jerk_lim = float(self.args.jerk_max)  # m/s^3
            da_max = jerk_lim * self.Ts
            a_prev = [0.0] * active_n

            for k in range(n_total):
                v_ref = float(v_ref_list[k])
                self._park_inactive(inactive_slots)

                for local_i in range(active_n):
                    s = self.slots[local_i]

                    self._update_dist(s)
                    if s.dist_m > float(self.args.max_dist_m):
                        total_cost[local_i] += 1e6  # penalty, but continue

                    v_long = get_v_long(s.actor)
                    _, _, a_meas = self._accel_filtered(s)

                    # Update speed-range selection for scheduled inner PID.
                    s.current_range, changed = update_speed_range(
                        s.current_range, abs(float(v_long)), self.v1, self.v2, self.h
                    )
                    if changed:
                        # reset the newly selected inner PID(s) to avoid windup transients
                        if s.current_range == "low":
                            s.inner_low.reset()
                            if s.inner_thr_low is not None:
                                s.inner_thr_low.reset()
                            if s.inner_brk_low is not None:
                                s.inner_brk_low.reset()
                        elif s.current_range == "mid":
                            s.inner_mid.reset()
                            if s.inner_thr_mid is not None:
                                s.inner_thr_mid.reset()
                            if s.inner_brk_mid is not None:
                                s.inner_brk_mid.reset()
                        else:
                            s.inner_high.reset()
                            if s.inner_thr_high is not None:
                                s.inner_thr_high.reset()
                            if s.inner_brk_high is not None:
                                s.inner_brk_high.reset()
                        s.mode = "coast"

                    # outer -> a_cmd (already saturated)
                    a_cmd = float(s.outer.step(v_ref, v_long))

                    # rate limit (jerk limit) -> a_des
                    a_des = a_prev[local_i] + clamp(a_cmd - a_prev[local_i], -da_max, +da_max)
                    a_prev[local_i] = a_des

                    # inner -> u (single or split)
                    if bool(self.args.split_inner):
                        # pick inner PID by (speed_range, sign(a_des))
                        if a_des >= 0.0:
                            if s.current_range == "low":
                                pid_thr = s.inner_thr_low
                            elif s.current_range == "mid":
                                pid_thr = s.inner_thr_mid
                            else:
                                pid_thr = s.inner_thr_high
                            if pid_thr is None:
                                raise RuntimeError("split_inner is set but throttle PID is missing")
                            u = float(pid_thr.step(a_des, a_meas))
                            u = clamp(u, 0.0, 1.0)
                        else:
                            if s.current_range == "low":
                                pid_brk = s.inner_brk_low
                            elif s.current_range == "mid":
                                pid_brk = s.inner_brk_mid
                            else:
                                pid_brk = s.inner_brk_high
                            if pid_brk is None:
                                raise RuntimeError("split_inner is set but brake PID is missing")
                            u = float(pid_brk.step(a_des, a_meas))
                            u = clamp(u, -1.0, 0.0)
                    else:
                        if s.current_range == "low":
                            pid = s.inner_low
                        elif s.current_range == "mid":
                            pid = s.inner_mid
                        else:
                            pid = s.inner_high
                        u = float(pid.step(a_des, a_meas))
                        u = clamp(u, -1.0, 1.0)
                    if abs(u) < float(self.args.u_deadband):
                        u = 0.0

                    # Hysteresis around zero effort to avoid chatter.
                    s.mode = mode_from_effort(s.mode, float(u), float(self.args.inner_db_on), float(self.args.inner_db_off))
                    thr = clamp(u, 0.0, 1.0) if s.mode == "throttle" else 0.0
                    brk = clamp(-u, 0.0, 1.0) if s.mode == "brake" else 0.0

                    steer = lane_keep_steer(
                        s.actor, self.carla_map,
                        lookahead_m=self.args.lookahead_m,
                        steer_gain=self.args.steer_gain,
                        wheelbase_m=self.args.wheelbase_m,
                    )
                    s.actor.apply_control(carla.VehicleControl(throttle=thr, brake=brk, steer=steer))

                    Vref[local_i][k] = v_ref
                    V[local_i][k] = v_long
                    Ades[local_i][k] = a_des
                    U[local_i][k] = u

                self.world.tick()
                self._render(f"{sc.name} v_ref={v_ref:.1f}")

            # compute scenario cost per particle
            for local_i in range(active_n):
                m = scenario_metrics(
                    Ts=self.Ts,
                    v_ref=Vref[local_i],
                    v=V[local_i],
                    a_des=Ades[local_i],
                    u=U[local_i],
                    dv=sc.dv,
                    settle_pct=float(self.args.settle_pct),
                    settle_abs_min=float(self.args.settle_abs_min),
                )
                track_term = float(self.args.w_track) * (m["IAE"] + float(self.args.sse_alpha) * m["SSE"])
                os_term = float(self.args.w_os) * m["OS"]
                settle_term = float(self.args.w_settle) * m["SETTLE"]
                a_term = float(self.args.w_a) * m["a_rms"]
                da_term = float(self.args.w_da) * m["da_rms"]
                sat_term = float(self.args.w_sat_u) * m["sat_u"]
                sw_term = float(self.args.w_switch) * m["switch_rate"]
                J = float(track_term + os_term + settle_term + a_term + da_term + sat_term + sw_term)
                total_cost[local_i] += float(J)
                if debug_breakdown is not None:
                    debug_breakdown[local_i].append(
                        {
                            "track": float(track_term),
                            "os": float(os_term),
                            "settle": float(settle_term),
                            "a_rms": float(a_term),
                            "da_rms": float(da_term),
                            "sat_u": float(sat_term),
                            "switch": float(sw_term),
                            "J": float(J),
                            "IAE": float(m["IAE"]),
                            "SSE": float(m["SSE"]),
                            "OS": float(m["OS"]),
                            "SETTLE": float(m["SETTLE"]),
                            "sat_u_raw": float(m["sat_u"]),
                            "switch_rate": float(m["switch_rate"]),
                        }
                    )

        if debug_breakdown is not None and active_n > 0:
            best_local = int(np.argmin(np.array(total_cost, dtype=np.float64)))
            best_pidx = int(particle_indices[best_local])
            print(f"[DEBUG][iter {it_idx} batch {batch_idx}] best P{best_pidx} J={total_cost[best_local]:.3f} x={fmt_vec(particles[best_pidx].x)}")
            for sc_i, sc in enumerate(scenarios):
                if sc_i >= len(debug_breakdown[best_local]):
                    continue
                d = debug_breakdown[best_local][sc_i]
                print(
                    f"  {sc.name:12s} J={d['J']:.3f} "
                    f"(track={d['track']:.3f} os={d['os']:.3f} settle={d['settle']:.3f} "
                    f"a={d['a_rms']:.3f} da={d['da_rms']:.3f} sat={d['sat_u']:.3f} sw={d['switch']:.3f}) "
                    f"| sat_u={d['sat_u_raw']:.3f} switch_rate={d['switch_rate']:.2f}"
                )

        return [float(x) for x in total_cost]


# --------------------------- main ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    # CARLA
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--vehicle-bp", type=str, default="vehicle.tesla.model3")

    # rendering (default off / fast)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--dashcam-vehicle", type=int, default=0)
    ap.add_argument("--cam-w", type=int, default=960)
    ap.add_argument("--cam-h", type=int, default=540)
    ap.add_argument("--cam-fov", type=int, default=90)

    ap.add_argument("--verbose", action="store_true")

    # PSO
    ap.add_argument("--particles", type=int, default=8)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--pso-w", type=float, default=0.72)
    ap.add_argument("--pso-c1", type=float, default=1.7)
    ap.add_argument("--pso-c2", type=float, default=1.7)

    # bounds for OUTER PID (defaults)
    ap.add_argument("--kp-min", type=float, default=float(OUTER_BOUNDS_DEFAULT["kp"][0]))
    ap.add_argument("--kp-max", type=float, default=float(OUTER_BOUNDS_DEFAULT["kp"][1]))
    ap.add_argument("--ki-min", type=float, default=float(OUTER_BOUNDS_DEFAULT["ki"][0]))
    ap.add_argument("--ki-max", type=float, default=float(OUTER_BOUNDS_DEFAULT["ki"][1]))
    ap.add_argument("--kd-min", type=float, default=float(OUTER_BOUNDS_DEFAULT["kd"][0]))
    ap.add_argument("--kd-max", type=float, default=float(OUTER_BOUNDS_DEFAULT["kd"][1]))
    ap.add_argument("--N-min", type=float, default=float(OUTER_BOUNDS_DEFAULT["N"][0]))
    ap.add_argument("--N-max", type=float, default=float(OUTER_BOUNDS_DEFAULT["N"][1]))

    # OUTER limits/shaping
    ap.add_argument("--a-max-throttle", type=float, default=2.5)  # +m/s^2
    ap.add_argument("--a-max-brake", type=float, default=5.0)     # magnitude, output min is -a_max_brake
    ap.add_argument("--jerk-max", type=float, default=4.0)        # m/s^3
    ap.add_argument("--outer-kb-aw", type=float, default=1.0)
    ap.add_argument("--outer-der-on-meas", action="store_true")

    # INNER fixed PID (defaults are hardcoded above; override via CLI if needed)
    ap.add_argument("--inner-kp", type=float, default=None)
    ap.add_argument("--inner-ki", type=float, default=None)
    ap.add_argument("--inner-kd", type=float, default=None)
    ap.add_argument("--inner-N", type=float, default=None)
    ap.add_argument("--inner-kb-aw", type=float, default=0.15)
    ap.add_argument("--inner-der-on-meas", action="store_true")
    ap.add_argument("--u-deadband", type=float, default=0.0)
    ap.add_argument("--inner-db-on", type=float, default=0.10, help="Inner actuator hysteresis ON threshold (|u|).")
    ap.add_argument("--inner-db-off", type=float, default=0.05, help="Inner actuator hysteresis OFF threshold (|u|).")

    # Optional split inner PIDs (throttle + brake) to avoid cross-regime coupling.
    # If enabled, the inner PIDs are selected based on sign(a_des).
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--split-inner",
        dest="split_inner",
        action="store_true",
        default=bool(SPLIT_INNER_DEFAULT),
        help="Use two inner PIDs: throttle (uin[0,1]) and brake (uin[-1,0]).",
    )
    g.add_argument(
        "--no-split-inner",
        dest="split_inner",
        action="store_false",
        help="Use a single inner PID (uin[-1,1]).",
    )

    # Optional scheduled inner PID overrides (per speed range).
    # Format: "kp,ki,kd,N". If omitted, uses the base args above.
    ap.add_argument("--inner-low", type=str, default=None)
    ap.add_argument("--inner-mid", type=str, default=None)
    ap.add_argument("--inner-high", type=str, default=None)
    ap.add_argument("--inner-thr-low", type=str, default=None)
    ap.add_argument("--inner-thr-mid", type=str, default=None)
    ap.add_argument("--inner-thr-high", type=str, default=None)
    ap.add_argument("--inner-brk-low", type=str, default=None)
    ap.add_argument("--inner-brk-mid", type=str, default=None)
    ap.add_argument("--inner-brk-high", type=str, default=None)

    # Throttle inner PID (defaults to --inner-* if not provided)
    ap.add_argument("--inner-thr-kp", type=float, default=None)
    ap.add_argument("--inner-thr-ki", type=float, default=None)
    ap.add_argument("--inner-thr-kd", type=float, default=None)
    ap.add_argument("--inner-thr-N", type=float, default=None)
    ap.add_argument("--inner-thr-kb-aw", type=float, default=None)
    ap.add_argument("--inner-thr-der-on-meas", action="store_true")

    # Brake inner PID (defaults to --inner-* if not provided)
    ap.add_argument("--inner-brk-kp", type=float, default=None)
    ap.add_argument("--inner-brk-ki", type=float, default=None)
    ap.add_argument("--inner-brk-kd", type=float, default=None)
    ap.add_argument("--inner-brk-N", type=float, default=None)
    ap.add_argument("--inner-brk-kb-aw", type=float, default=None)
    ap.add_argument("--inner-brk-der-on-meas", action="store_true")

    # accel estimation
    ap.add_argument("--accel-lpf-alpha", type=float, default=0.35)

    # lane keep
    ap.add_argument("--lookahead-m", type=float, default=12.0)
    ap.add_argument("--steer-gain", type=float, default=1.0)
    ap.add_argument("--wheelbase-m", type=float, default=2.8)

    # warmup (old speed PID)
    ap.add_argument("--warm-max-time-s", type=float, default=15.0)
    ap.add_argument("--warm-band", type=float, default=1.5)
    ap.add_argument("--warm-hold-s", type=float, default=1.0)
    ap.add_argument("--warm-v1", type=float, default=11.11)
    ap.add_argument("--warm-v2", type=float, default=22.22)
    ap.add_argument("--warm-hyst", type=float, default=1.0)
    ap.add_argument("--warm-kb-aw", type=float, default=1.0)

    # safety reset
    ap.add_argument("--max-dist-m", type=float, default=250.0)

    # scenarios
    ap.add_argument("--t-pre", type=float, default=0.8)
    ap.add_argument("--t-step", type=float, default=4.0)
    ap.add_argument("--t-post", type=float, default=1.2)
    ap.add_argument("--dv-small", type=float, default=2.0)
    ap.add_argument("--dv-large", type=float, default=6.0)
    ap.add_argument("--vop-low", type=float, default=10.0)
    ap.add_argument("--vop-high", type=float, default=20.0)
    ap.add_argument("--dv-brake-strong", type=float, default=-10.0)

    # settle params
    ap.add_argument("--settle-pct", type=float, default=0.05)
    ap.add_argument("--settle-abs-min", type=float, default=0.2)

    # cost weights
    ap.add_argument("--w-track", type=float, default=4.0)
    ap.add_argument("--sse-alpha", type=float, default=0.15)
    ap.add_argument("--w-os", type=float, default=3.0)
    ap.add_argument("--w-settle", type=float, default=2.0)
    ap.add_argument("--w-a", type=float, default=0.15)
    ap.add_argument("--w-da", type=float, default=0.25)
    ap.add_argument("--w-sat-u", type=float, default=3.0)
    ap.add_argument("--w-switch", type=float, default=1.0)

    # debug prints
    ap.add_argument(
        "--debug-metrics",
        action="store_true",
        help="Print per-scenario cost breakdown for the best particle in each batch (useful to see what dominates J).",
    )

    args = ap.parse_args()

    if args.render and cv2 is None:
        print("[WARN] --render set but opencv not installed; run `pip install opencv-python`.")

    # Track whether base gains were provided via CLI before filling defaults.
    base_inner_cli = any(v is not None for v in (args.inner_kp, args.inner_ki, args.inner_kd, args.inner_N))
    base_thr_cli = any(v is not None for v in (args.inner_thr_kp, args.inner_thr_ki, args.inner_thr_kd, args.inner_thr_N))
    base_brk_cli = any(v is not None for v in (args.inner_brk_kp, args.inner_brk_ki, args.inner_brk_kd, args.inner_brk_N))

    # If base inner PID wasn't provided, default to the hardcoded scheduled gains.
    if args.inner_kp is None:
        args.inner_kp = float(INNER_SINGLE_GAINS["mid"][0])
    if args.inner_ki is None:
        args.inner_ki = float(INNER_SINGLE_GAINS["mid"][1])
    if args.inner_kd is None:
        args.inner_kd = float(INNER_SINGLE_GAINS["mid"][2])
    if args.inner_N is None:
        args.inner_N = float(INNER_SINGLE_GAINS["mid"][3])

    # Fill split-inner defaults from the base inner PID if not explicitly set.
    if args.inner_thr_kp is None:
        args.inner_thr_kp = float(args.inner_kp)
    if args.inner_thr_ki is None:
        args.inner_thr_ki = float(args.inner_ki)
    if args.inner_thr_kd is None:
        args.inner_thr_kd = float(args.inner_kd)
    if args.inner_thr_N is None:
        args.inner_thr_N = float(args.inner_N)
    if args.inner_thr_kb_aw is None:
        args.inner_thr_kb_aw = float(args.inner_kb_aw)
    if not bool(args.inner_thr_der_on_meas):
        args.inner_thr_der_on_meas = bool(args.inner_der_on_meas)

    if args.inner_brk_kp is None:
        args.inner_brk_kp = float(args.inner_kp)
    if args.inner_brk_ki is None:
        args.inner_brk_ki = float(args.inner_ki)
    if args.inner_brk_kd is None:
        args.inner_brk_kd = float(args.inner_kd)
    if args.inner_brk_N is None:
        args.inner_brk_N = float(args.inner_N)
    if args.inner_brk_kb_aw is None:
        args.inner_brk_kb_aw = float(args.inner_kb_aw)
    if not bool(args.inner_brk_der_on_meas):
        args.inner_brk_der_on_meas = bool(args.inner_der_on_meas)

    # If per-range strings were not provided, populate them from either:
    # - base CLI gains (if provided), else
    # - hardcoded scheduled tables above.
    def _pid4_str(*vals: float) -> str:
        kp, ki, kd, N = (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
        return f"{kp},{ki},{kd},{N}"

    if args.inner_low is None and args.inner_mid is None and args.inner_high is None:
        if base_inner_cli:
            args.inner_low = _pid4_str(args.inner_kp, args.inner_ki, args.inner_kd, args.inner_N)
            args.inner_mid = args.inner_low
            args.inner_high = args.inner_low
        else:
            args.inner_low = _pid4_str(*INNER_SINGLE_GAINS["low"])
            args.inner_mid = _pid4_str(*INNER_SINGLE_GAINS["mid"])
            args.inner_high = _pid4_str(*INNER_SINGLE_GAINS["high"])

    if args.inner_thr_low is None and args.inner_thr_mid is None and args.inner_thr_high is None:
        if base_thr_cli:
            args.inner_thr_low = _pid4_str(args.inner_thr_kp, args.inner_thr_ki, args.inner_thr_kd, args.inner_thr_N)
            args.inner_thr_mid = args.inner_thr_low
            args.inner_thr_high = args.inner_thr_low
        else:
            args.inner_thr_low = _pid4_str(*INNER_THROTTLE_GAINS["low"])
            args.inner_thr_mid = _pid4_str(*INNER_THROTTLE_GAINS["mid"])
            args.inner_thr_high = _pid4_str(*INNER_THROTTLE_GAINS["high"])

    if args.inner_brk_low is None and args.inner_brk_mid is None and args.inner_brk_high is None:
        if base_brk_cli:
            args.inner_brk_low = _pid4_str(args.inner_brk_kp, args.inner_brk_ki, args.inner_brk_kd, args.inner_brk_N)
            args.inner_brk_mid = args.inner_brk_low
            args.inner_brk_high = args.inner_brk_low
        else:
            args.inner_brk_low = _pid4_str(*INNER_BRAKE_GAINS["low"])
            args.inner_brk_mid = _pid4_str(*INNER_BRAKE_GAINS["mid"])
            args.inner_brk_high = _pid4_str(*INNER_BRAKE_GAINS["high"])

    n_particles = int(args.particles)
    if n_particles < 1:
        raise SystemExit("--particles must be >= 1")

    bounds = [
        (float(args.kp_min), float(args.kp_max)),
        (float(args.ki_min), float(args.ki_max)),
        (float(args.kd_min), float(args.kd_max)),
        (float(args.N_min), float(args.N_max)),
    ]

    scenarios = [
        ScenarioSpec("S1_up_small",  v_op=float(args.vop_low),  dv=+float(args.dv_small),  t_pre=args.t_pre, t_step=args.t_step, t_post=args.t_post),
        ScenarioSpec("S2_up_large",  v_op=float(args.vop_low),  dv=+float(args.dv_large),  t_pre=args.t_pre, t_step=args.t_step, t_post=args.t_post),
        ScenarioSpec("S3_dn_small",  v_op=float(args.vop_low),  dv=-float(args.dv_small),  t_pre=args.t_pre, t_step=args.t_step, t_post=args.t_post),
        ScenarioSpec("S4_dn_strong", v_op=float(args.vop_high), dv=float(args.dv_brake_strong), t_pre=args.t_pre, t_step=args.t_step, t_post=args.t_post),
    ]

    particles = pso_init(bounds=bounds, n_particles=n_particles, seed=int(args.seed))
    gbest_x = particles[0].x.copy()
    gbest_J = float("inf")

    tuner = CarlaOuterTuner(args)

    try:
        batch_size = 8
        n_batches = int(math.ceil(n_particles / batch_size))

        print(f"[INIT] particles={n_particles} iters={args.iters} batches/iter={n_batches}")
        if bool(args.split_inner):
            print("[INIT] inner fixed: SPLIT throttle+brake")
            print(
                f"  thr: kp={args.inner_thr_kp} ki={args.inner_thr_ki} kd={args.inner_thr_kd} N={args.inner_thr_N} "
                f"kb_aw={args.inner_thr_kb_aw} der_on_meas={bool(args.inner_thr_der_on_meas)} out=[0,1]"
            )
            print(
                f"  brk: kp={args.inner_brk_kp} ki={args.inner_brk_ki} kd={args.inner_brk_kd} N={args.inner_brk_N} "
                f"kb_aw={args.inner_brk_kb_aw} der_on_meas={bool(args.inner_brk_der_on_meas)} out=[-1,0]"
            )
        else:
            print(f"[INIT] inner fixed: SINGLE kp={args.inner_kp} ki={args.inner_ki} kd={args.inner_kd} N={args.inner_N} out=[-1,1]")
        print(f"[INIT] render={args.render} (default fast: no_rendering_mode={not args.render})")

        for it in range(int(args.iters)):
            print("\n" + "=" * 110)
            print(f"[PSO] ITER {it+1}/{args.iters}")

            all_costs = [float("inf")] * n_particles

            for b in range(n_batches):
                start = b * batch_size
                end = min(n_particles, (b + 1) * batch_size)
                idxs = list(range(start, end))

                print(f"[BATCH] {b+1}/{n_batches} particles {start}..{end-1} (n={len(idxs)})")
                costs = tuner.evaluate_batch(
                    particles=particles,
                    particle_indices=idxs,
                    scenarios=scenarios,
                    it_idx=it+1,
                    batch_idx=b+1,
                )
                for local_i, p_idx in enumerate(idxs):
                    all_costs[p_idx] = float(costs[local_i])

                print(f"[BATCH] best J in batch: {min(all_costs[p] for p in idxs):.3f}")

            # update pbest / gbest
            for i, p in enumerate(particles):
                J = float(all_costs[i])
                if J < p.pbest_J:
                    p.pbest_J = J
                    p.pbest_x = p.x.copy()
                if J < gbest_J:
                    gbest_J = J
                    gbest_x = p.x.copy()

            # report
            order = np.argsort(np.array(all_costs))
            topk = min(10, n_particles)
            print("[ITER SUMMARY] top particles")
            for r in range(topk):
                i = int(order[r])
                print(f"  #{r+1:02d} P{i:03d} J={all_costs[i]:12.3f} x={fmt_vec(particles[i].x)}")
            print(f"[GBEST] J={gbest_J:.6f} x={fmt_vec(gbest_x)}")

            # PSO step
            pso_update(
                particles=particles,
                gbest_x=gbest_x,
                bounds=bounds,
                w=float(args.pso_w),
                c1=float(args.pso_c1),
                c2=float(args.pso_c2),
                seed=int(args.seed) + 1000 + it,
            )

        print("\n" + "=" * 110)
        print("[DONE]")
        print(f"Best cost: {gbest_J:.6f}")
        print(f"Best OUTER params: {fmt_vec(gbest_x)}")

    finally:
        tuner.destroy()


if __name__ == "__main__":
    main()
