#!/usr/bin/env python3
"""
test_inner_pid_single_vehicle_multi_trials_csv.py

Single vehicle, Town04, dt=0.01 (substep 0.001).

What it does (requested):
- Spawn 1 vehicle (any spawnpoint index you choose).
- For EACH trial:
  1) Teleport vehicle back to spawn (also if traveled > 250m since last TP).
  2) Warm-up to operating speed v_op using OLD scheduled speed PID (from PID.py).
     Wait until within a generous band for hold time.
  3) Run an INNER PID acceleration trial for a_des != 0 (step for t_step seconds),
     with pre/post zeros.
  4) Log everything to a single CSV with a "trial_id" column.
- Does 10 different a_des values (default list) around the operating point.
- After each trial OR if distance>250m, teleports.

CSV columns:
  trial_id, phase, t, t_trial, a_des, a_raw, a_filt, u, throttle, brake, v_long, steer, dist_m, x, y, z

Phases:
  "WARMUP" or "TEST"

Optional:
- --realtime to pace to walltime 0.01 s per tick (for real-time viewing).
- OpenCV camera window (chase cam) if opencv-python installed.
"""

import argparse
import csv
import math
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import carla

from PID import PID

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


# ----------------------------- helpers -----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def loc_str(l: carla.Location) -> str:
    return f"({l.x:.1f},{l.y:.1f},{l.z:.1f})"


def get_v_long(vehicle: carla.Vehicle) -> float:
    v = vehicle.get_velocity()
    fwd = vehicle.get_transform().get_forward_vector()
    return float(v.x * fwd.x + v.y * fwd.y + v.z * fwd.z)


def lane_keep_steer(vehicle: carla.Vehicle, carla_map: carla.Map,
                    lookahead_m: float = 12.0, steer_gain: float = 1.0, wheelbase_m: float = 2.8) -> float:
    tf = vehicle.get_transform()
    wp = carla_map.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return 0.0
    nxt = wp.next(max(1.0, float(lookahead_m)))
    if not nxt:
        return 0.0

    target = nxt[0].transform.location
    vec = target - tf.location

    fwd = tf.get_forward_vector()
    right = tf.get_right_vector()

    x = vec.x * fwd.x + vec.y * fwd.y + vec.z * fwd.z
    y = vec.x * right.x + vec.y * right.y + vec.z * right.z
    if x <= 1e-3:
        return 0.0

    curvature = (2.0 * y) / (x * x + y * y)
    steer = math.atan(wheelbase_m * curvature)
    return clamp(steer_gain * steer, -1.0, 1.0)


@dataclass
class CameraState:
    sensor: carla.Sensor
    last_image: Optional[np.ndarray] = None


# ----------------------------- runner -----------------------------

class MultiTrialInnerPIDTester:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.Ts = 0.01

        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(30.0)

        print("[CARLA] Loading Town04 ...")
        self.client.load_world("Town04")
        time.sleep(1.2)

        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()

        self.orig_settings = self.world.get_settings()
        self._apply_settings()

        self.vehicle: Optional[carla.Vehicle] = None
        self.camera: Optional[CameraState] = None
        self.last_img = None
        self.collided = False

        self.spawn_tf: Optional[carla.Transform] = None

        self.v_prev = 0.0
        self.a_filt = 0.0

        self.last_loc: Optional[carla.Location] = None
        self.dist_m = 0.0

        self._spawn_vehicle()
        self._attach_sensors()

    # ---------------- CARLA setup ----------------

    def _apply_settings(self) -> None:
        s = self.world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = self.Ts
        s.substepping = True
        s.max_substep_delta_time = 0.001
        s.max_substeps = 10
        s.no_rendering_mode = False  # camera visible
        self.world.apply_settings(s)
        print(f"[CARLA] sync ON dt={self.Ts}, substep=0.001, no_render=False")

    def destroy(self) -> None:
        print("[CARLA] Destroying actors...")
        try:
            if self.camera is not None:
                self.camera.sensor.stop()
                self.camera.sensor.destroy()
        except Exception:
            pass
        try:
            if self.vehicle is not None:
                self.vehicle.destroy()
        except Exception:
            pass
        try:
            self.world.apply_settings(self.orig_settings)
        except Exception:
            pass
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _spawn_vehicle(self) -> None:
        spawns = self.map.get_spawn_points()
        sp = spawns[int(self.args.spawn_index)]
        sp.location.z += 0.35
        self.spawn_tf = sp

        bp = self.bp_lib.find(self.args.vehicle_bp)
        self.vehicle = self.world.try_spawn_actor(bp, sp)
        if self.vehicle is None:
            raise RuntimeError(f"Failed to spawn vehicle at spawn index {self.args.spawn_index}")
        self.vehicle.set_autopilot(False)

        self.last_loc = self.vehicle.get_transform().location
        self.dist_m = 0.0

        print(f"[CARLA] Vehicle spawned at spawn[{self.args.spawn_index}] loc={loc_str(sp.location)} yaw={sp.rotation.yaw:.1f}")

        for _ in range(10):
            self._tick()

    def _attach_sensors(self) -> None:
        if cv2 is None:
            print("[WARN] cv2 not available; no camera window. Install opencv-python.")
            return
        assert self.vehicle is not None

        cam_bp = self.bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.args.cam_w))
        cam_bp.set_attribute("image_size_y", str(self.args.cam_h))
        cam_bp.set_attribute("fov", str(self.args.cam_fov))

        cam_tf = carla.Transform(carla.Location(x=-6.5, z=2.5), carla.Rotation(pitch=-10.0))
        cam = self.world.spawn_actor(cam_bp, cam_tf, attach_to=self.vehicle)

        st = CameraState(sensor=cam, last_image=None)

        def _on_img(img: carla.Image):
            arr = np.frombuffer(img.raw_data, dtype=np.uint8)
            st.last_image = arr.reshape((img.height, img.width, 4))[:, :, :3]

        cam.listen(_on_img)
        self.camera = st

        print("[CAM] Chase camera attached.")

    def _render(self, overlay: str = "") -> None:
        if cv2 is None or self.camera is None or self.camera.last_image is None:
            return
        img = self.camera.last_image.copy()
        if overlay:
            cv2.putText(img, overlay, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("inner PID multi-trial test", img)
        cv2.waitKey(1)

    def _tick(self, overlay: str = "") -> None:
        t0 = time.perf_counter()
        self.world.tick()
        self._render(overlay)
        if self.args.realtime:
            elapsed = time.perf_counter() - t0
            sleep_s = self.Ts - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _update_dist(self) -> None:
        assert self.vehicle is not None and self.last_loc is not None
        loc = self.vehicle.get_transform().location
        dx = loc.x - self.last_loc.x
        dy = loc.y - self.last_loc.y
        dz = loc.z - self.last_loc.z
        self.dist_m += math.sqrt(dx * dx + dy * dy + dz * dz)
        self.last_loc = loc

    def _accel_filtered(self) -> Tuple[float, float, float]:
        assert self.vehicle is not None
        v_long = get_v_long(self.vehicle)
        a_raw = (v_long - self.v_prev) / self.Ts
        self.v_prev = v_long
        alpha = float(self.args.accel_lpf_alpha)
        self.a_filt = alpha * a_raw + (1.0 - alpha) * self.a_filt
        return v_long, a_raw, self.a_filt

    def _tp_to_spawn(self, reason: str) -> None:
        assert self.vehicle is not None and self.spawn_tf is not None
        print(f"[TP] Teleport to spawn. Reason: {reason}")
        self.vehicle.set_transform(self.spawn_tf)
        self.vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self.vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
        self.v_prev = 0.0
        self.a_filt = 0.0
        self.last_loc = self.vehicle.get_transform().location
        self.dist_m = 0.0
        for _ in range(10):
            self._tick(overlay="TP reset")

    # ---------------- warm-up with old scheduled PID ----------------

    def warmup_to_speed(self, csv_writer, trial_id: int, t_global: float) -> Tuple[bool, float]:
        """
        Warm-up with old speed PID scheduling.
        Logs to CSV during warmup with phase=WARMUP.
        Returns (ok, new_t_global).
        """
        assert self.vehicle is not None

        Ts = self.Ts
        u_min, u_max = -1.0, 1.0
        kb_aw = float(self.args.warm_kb_aw)

        # Old gains
        slow_pid = PID(0.1089194,  0.04906409, 0.0,  7.71822214, Ts, u_min, u_max, kb_aw, der_on_meas=True)
        mid_pid  = PID(0.11494122, 0.0489494,  0.0,  5.0,        Ts, u_min, u_max, kb_aw, der_on_meas=True)
        fast_pid = PID(0.19021246, 0.15704399, 0.0, 12.70886655, Ts, u_min, u_max, kb_aw, der_on_meas=True)
        slow_pid.reset(); mid_pid.reset(); fast_pid.reset()

        v1 = float(self.args.warm_v1)
        v2 = float(self.args.warm_v2)
        h = float(self.args.warm_hyst)

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

        v_op = float(self.args.v_op)
        band = float(self.args.warm_band)
        hold_ticks_req = int(round(float(self.args.warm_hold_s) / Ts))
        hold_ticks = 0
        max_steps = int(round(float(self.args.warm_max_time_s) / Ts))
        range_name = "low"

        if self.args.live_log:
            print(f"[WARMUP] v_op={v_op:.2f} band=±{band:.2f} hold={self.args.warm_hold_s:.2f}s")

        t_trial = 0.0
        for k in range(max_steps):
            self._update_dist()
            if self.dist_m > float(self.args.max_dist_m):
                print(f"[WARMUP] distance>{self.args.max_dist_m}m -> TP needed")
                return False, t_global

            v_long = get_v_long(self.vehicle)
            range_new = update_range(range_name, v_long)
            if range_new != range_name:
                range_name = range_new
                select_pid(range_name).reset()

            pid = select_pid(range_name)
            u = float(pid.step(v_op, v_long))
            thr = max(0.0, u)
            brk = max(0.0, -u)

            steer = lane_keep_steer(self.vehicle, self.map,
                                    lookahead_m=self.args.lookahead_m,
                                    steer_gain=self.args.steer_gain,
                                    wheelbase_m=self.args.wheelbase_m)

            self.vehicle.apply_control(carla.VehicleControl(throttle=thr, brake=brk, steer=steer))

            # accel signals for logging (not used by warmup PID)
            if k == 0:
                self.v_prev = v_long
            v_long2, a_raw, a_filt = self._accel_filtered()

            # log
            tf = self.vehicle.get_transform()
            csv_writer.writerow([
                trial_id, "WARMUP", t_global, t_trial,
                "",  # a_des empty
                a_raw, a_filt, u, thr, brk, v_long2, steer, self.dist_m,
                tf.location.x, tf.location.y, tf.location.z
            ])

            in_band = abs(v_long - v_op) <= band
            hold_ticks = hold_ticks + 1 if in_band else 0

            if self.args.live_log and (k % int(self.args.log_every) == 0):
                print(f"[WARMUP][t={t_trial:5.2f}] v={v_long:5.2f} range={range_name:>4} "
                      f"u={u:+.3f} in_band={int(in_band)} hold={hold_ticks}/{hold_ticks_req} dist={self.dist_m:6.1f}")

            self._tick(overlay=f"WARMUP v={v_long:5.2f}/{v_op:5.2f} range={range_name}")
            t_global += Ts
            t_trial += Ts

            if hold_ticks >= hold_ticks_req:
                if self.args.live_log:
                    print("[WARMUP] OK")
                return True, t_global

        print("[WARMUP] timeout")
        return False, t_global

    # ---------------- inner PID trial ----------------

    def run_trial(self, trial_id: int, a_step: float, csv_writer, t_global: float) -> float:
        """
        Run one acceleration trial with INNER PID.
        Logs phase=TEST.
        Returns new t_global.
        """
        assert self.vehicle is not None

        inner = PID(
            float(self.args.kp),
            float(self.args.ki),
            float(self.args.kd),
            float(self.args.N),
            self.Ts,
            -1.0, 1.0,
            float(self.args.kb_aw),
            der_on_meas=bool(self.args.inner_der_on_meas),
            prop_on_meas=False,
            derivative_disc_method="tustin",
            integral_disc_method="backeuler",
        )
        inner.reset()

        # re-init accel estimator at trial start
        self.v_prev = get_v_long(self.vehicle)
        self.a_filt = 0.0

        # profile: 0 -> step -> 0
        prof: List[Tuple[float, float]] = [
            (float(self.args.t_pre), 0.0),
            (float(self.args.t_step), float(a_step)),
            (float(self.args.t_post), 0.0),
        ]
        timeline: List[float] = []
        for dur, a in prof:
            timeline += [a] * int(round(dur / self.Ts))

        if self.args.live_log:
            print(f"[TRIAL {trial_id}] a_step={a_step:+.2f}  (pre={self.args.t_pre}s step={self.args.t_step}s post={self.args.t_post}s)")

        t_trial = 0.0
        for k, a_des in enumerate(timeline):
            self._update_dist()
            if self.dist_m > float(self.args.max_dist_m):
                print(f"[TEST] distance>{self.args.max_dist_m}m -> TP needed")
                break

            v_long, a_raw, a_filt = self._accel_filtered()

            u = float(inner.step(a_des, a_filt))
            if abs(u) < float(self.args.u_deadband):
                u = 0.0

            thr = max(0.0, u)
            brk = max(0.0, -u)

            steer = lane_keep_steer(self.vehicle, self.map,
                                    lookahead_m=self.args.lookahead_m,
                                    steer_gain=self.args.steer_gain,
                                    wheelbase_m=self.args.wheelbase_m)

            self.vehicle.apply_control(carla.VehicleControl(throttle=thr, brake=brk, steer=steer))

            tf = self.vehicle.get_transform()
            csv_writer.writerow([
                trial_id, "TEST", t_global, t_trial,
                a_des,
                a_raw, a_filt, u, thr, brk, v_long, steer, self.dist_m,
                tf.location.x, tf.location.y, tf.location.z
            ])

            if self.args.live_log and (k % int(self.args.log_every) == 0):
                print(f"  [TEST t={t_trial:5.2f}] a_des={a_des:+.2f} a={a_filt:+.2f} u={u:+.3f} v={v_long:5.2f} dist={self.dist_m:6.1f}")

            self._tick(overlay=f"TRIAL {trial_id} a_des={a_des:+.2f} a={a_filt:+.2f} u={u:+.2f} v={v_long:.2f}")
            t_global += self.Ts
            t_trial += self.Ts

        # settle briefly neutral
        for _ in range(int(round(0.3 / self.Ts))):
            steer = lane_keep_steer(self.vehicle, self.map,
                                    lookahead_m=self.args.lookahead_m,
                                    steer_gain=self.args.steer_gain,
                                    wheelbase_m=self.args.wheelbase_m)
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=steer))
            self._tick(overlay=f"TRIAL {trial_id} DONE")
            t_global += self.Ts

        return t_global

    # ---------------- master run ----------------

    def run(self) -> None:
        # 10 a_des values (requested). Can override from CLI with --a-steps
        if self.args.a_steps:
            a_steps = [float(x) for x in self.args.a_steps.split(",")]
        else:
            a_steps = [-3.0, -2.5, -2.0, -1.5, -1.0, +0.8, +1.2, +1.6, +2.0, +2.4]

        if len(a_steps) != 10:
            print(f"[WARN] a_steps has {len(a_steps)} values (expected 10). Proceeding anyway.")

        csv_name = self.args.out_csv
        if not csv_name:
            csv_name = (
                f"inner_pid_multi_trials_kp{self.args.kp:.3f}_"
                f"ki{self.args.ki:.3f}_"
                f"kd{self.args.kd:.3f}_"
                f"N{self.args.N:.1f}_"
                f"vop{self.args.v_op:.1f}.csv"
            )

        print(f"[CSV] Writing: {csv_name}")
        print(f"[PID] kp={self.args.kp} ki={self.args.ki} kd={self.args.kd} N={self.args.N} der_on_meas={self.args.inner_der_on_meas}")
        print(f"[PLAN] trials={len(a_steps)} | TP after each trial OR if dist>{self.args.max_dist_m}m")

        t_global = 0.0

        with open(csv_name, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow([
                "trial_id", "phase", "t", "t_trial",
                "a_des",
                "a_raw", "a_filt", "u", "throttle", "brake", "v_long", "steer", "dist_m",
                "x", "y", "z"
            ])

            for trial_id, a_step in enumerate(a_steps):
                # Always TP at start of each trial (requested)
                self._tp_to_spawn(reason=f"start trial {trial_id}")

                # Warm-up to v_op (old PID)
                ok, t_global = self.warmup_to_speed(wr, trial_id, t_global)
                if not ok:
                    # if warmup failed because distance exceeded, TP and retry once
                    self._tp_to_spawn(reason=f"warmup fail trial {trial_id} -> retry")
                    ok, t_global = self.warmup_to_speed(wr, trial_id, t_global)
                    if not ok:
                        print(f"[TRIAL {trial_id}] warmup failed twice, skipping trial.")
                        continue

                # Run inner trial
                t_global = self.run_trial(trial_id, a_step, wr, t_global)

                # TP at end of trial (requested)
                self._tp_to_spawn(reason=f"end trial {trial_id}")

        print("[DONE] All trials completed. CSV saved.")


# ----------------------------- CLI -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    # CARLA
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--vehicle-bp", type=str, default="vehicle.tesla.model3")
    ap.add_argument("--spawn-index", type=int, default=115)

    # Camera
    ap.add_argument("--cam-w", type=int, default=960)
    ap.add_argument("--cam-h", type=int, default=540)
    ap.add_argument("--cam-fov", type=int, default=90)

    # Real-time pacing
    ap.add_argument("--realtime", action="store_true", help="Pace to walltime ~0.01s per tick")

    # Logging
    ap.add_argument("--live-log", action="store_true")
    ap.add_argument("--log-every", type=int, default=20, help="Ticks between console logs (20=0.2s)")

    # Lane keep
    ap.add_argument("--lookahead-m", type=float, default=12.0)
    ap.add_argument("--steer-gain", type=float, default=1.0)
    ap.add_argument("--wheelbase-m", type=float, default=2.8)

    # Warm-up params
    ap.add_argument("--v-op", type=float, default=10.0)
    ap.add_argument("--warm-max-time-s", type=float, default=15.0)
    ap.add_argument("--warm-band", type=float, default=1.5)
    ap.add_argument("--warm-hold-s", type=float, default=1.0)
    ap.add_argument("--warm-v1", type=float, default=11.11)
    ap.add_argument("--warm-v2", type=float, default=22.22)
    ap.add_argument("--warm-hyst", type=float, default=1.0)
    ap.add_argument("--warm-kb-aw", type=float, default=1.0)

    # Inner PID under test
    ap.add_argument("--kp", type=float, required=True)
    ap.add_argument("--ki", type=float, required=True)
    ap.add_argument("--kd", type=float, required=True)
    ap.add_argument("--N", type=float, required=True)
    ap.add_argument("--kb-aw", type=float, default=0.15)
    ap.add_argument("--inner-der-on-meas", action="store_true")
    ap.add_argument("--u-deadband", type=float, default=0.0)

    # Accel estimator
    ap.add_argument("--accel-lpf-alpha", type=float, default=0.35)

    # Trial profile for each a_step: 0 -> a_step -> 0
    ap.add_argument("--t-pre", type=float, default=0.6)
    ap.add_argument("--t-step", type=float, default=2.0)
    ap.add_argument("--t-post", type=float, default=1.0)

    # Trials set
    ap.add_argument("--a-steps", type=str, default="",
                    help="Comma-separated list of a_step values (m/s^2). If empty, uses a default set of 10.")
    ap.add_argument("--out-csv", type=str, default="", help="Output CSV filename")

    # TP / safety
    ap.add_argument("--max-dist-m", type=float, default=250.0)

    args = ap.parse_args()

    if cv2 is None:
        print("[WARN] cv2 not available; install opencv-python for camera window.")

    tester = MultiTrialInnerPIDTester(args)
    try:
        tester.run()
    finally:
        tester.destroy()


if __name__ == "__main__":
    main()
