#!/usr/bin/env python3
"""
PSO-based tuning of the *inner* accel PID directly in CARLA, using 8 vehicles
evaluated in parallel (one particle per vehicle).

Reference: pid_tuning_absolute.py (PSO structure + logging style), but the plant
is CARLA instead of a fitted model.

Key features:
- Fixed particle count = 8 (hardcoded swarm).
- Uses Town04 and spawns vehicles at a fixed list of spawnpoint indices.
- Runs synchronous mode with Ts=0.01.
- Accel measurement is computed as d/dt of planar speed with IIR low-pass filter.
- Inner-only tuning: desired accel schedule (both + and -) -> effort -> throttle/brake.

Notes:
- One provided spawnpoint index appears duplicated (81). This script will
  automatically offset duplicate transforms forward so vehicles don't overlap.
- For maximum speed, use --no-rendering.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass

from PID import PID

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    _HAVE_CV2 = True
except Exception:
    cv2 = None
    np = None
    _HAVE_CV2 = False

import queue
import threading


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def _planar_speed(vel) -> float:
    return float(math.sqrt(float(vel.x) ** 2 + float(vel.y) ** 2))


def _lowpass(prev: float, raw: float, beta: float) -> float:
    return float(prev + float(beta) * (float(raw) - float(prev)))


def _mode_from_effort(mode: str, u_effort: float, db_on: float, db_off: float) -> str:
    if mode == "coast":
        if u_effort > db_on:
            return "throttle"
        if u_effort < -db_on:
            return "brake"
        return "coast"
    if mode == "throttle":
        return "coast" if u_effort < db_off else "throttle"
    return "coast" if u_effort > -db_off else "brake"


@dataclass
class ParticleState:
    # parameters
    x: list[float]  # [kp, ki, kd, N]
    v: list[float]  # velocity in parameter space
    # bests
    pbest_x: list[float]
    pbest_J: float
    # runtime state for evaluation
    mode: str
    prev_u: float
    last_speed: float
    accel_filt: float
    aborted: bool
    abort_reason: str
    # accumulators
    sum_abs_aerr_pos: float
    sum_abs_aerr_neg: float
    n_pos: int
    n_neg: int
    sum_abs_u: float
    sum_abs_du: float
    sat_hits: int
    samples: int

    def reset_runtime(self) -> None:
        self.mode = "coast"
        self.prev_u = 0.0
        self.last_speed = 0.0
        self.accel_filt = 0.0
        self.aborted = False
        self.abort_reason = ""
        self.sum_abs_aerr_pos = 0.0
        self.sum_abs_aerr_neg = 0.0
        self.n_pos = 0
        self.n_neg = 0
        self.sum_abs_u = 0.0
        self.sum_abs_du = 0.0
        self.sat_hits = 0
        self.samples = 0


def _build_accel_schedule(args) -> list[tuple[float, float]]:
    sched: list[tuple[float, float]] = []
    for _ in range(int(args.cycles)):
        sched.extend(
            [
                (float(args.hold_s), float(args.a_pos)),
                (float(args.rest_s), 0.0),
                (float(args.hold_s), float(args.a_neg)),
                (float(args.rest_s), 0.0),
            ]
        )
    return sched


def _spawn_vehicles(world, bp_library, spawn_points, spawn_indices: list[int], blueprint_id: str):
    import carla  # type: ignore

    bp = bp_library.find(blueprint_id)
    vehicles = []
    used = {}
    for idx in spawn_indices:
        if not (0 <= int(idx) < len(spawn_points)):
            raise ValueError(f"Spawn index {idx} out of range (0..{len(spawn_points)-1})")

        base_tf = spawn_points[int(idx)]
        tf = carla.Transform(base_tf.location, base_tf.rotation)

        # If duplicate index, offset along forward vector to avoid overlap.
        used_count = used.get(int(idx), 0)
        if used_count > 0:
            fwd = tf.get_forward_vector()
            tf.location = tf.location + fwd * float(10.0 * used_count)
        used[int(idx)] = used_count + 1

        # Snap z to ground via spawn point z (slightly lowered).
        tf.location.z = float(base_tf.location.z) - 0.3

        veh = world.try_spawn_actor(bp, tf)
        if veh is None:
            # retry with small Z offsets
            ok = None
            for dz in [0.0, 0.2, 0.5, 1.0]:
                tf2 = carla.Transform(
                    carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + dz),
                    tf.rotation,
                )
                ok = world.try_spawn_actor(bp, tf2)
                if ok is not None:
                    veh = ok
                    break
            if veh is None:
                raise RuntimeError(f"Failed to spawn vehicle at spawnpoint {idx}")

        veh.set_autopilot(False)
        vehicles.append(veh)

    return vehicles


def main() -> None:
    parser = argparse.ArgumentParser(description="PSO tune inner accel PID in CARLA with 8 vehicles (8 particles).")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", type=str, default="Town04")
    parser.add_argument("--blueprint", type=str, default="vehicle.tesla.model3")
    parser.add_argument("--seed", type=int, default=1)

    # Fixed swarm spawn points (user-provided)
    parser.add_argument(
        "--spawn-indices",
        type=str,
        default="115,116,117,118,79,80,81,82",
        help="Comma-separated list of 8 spawn point indices.",
    )

    # Simulation
    parser.add_argument("--ts", type=float, default=0.01)
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument("--substepping", action="store_true", default=True)
    parser.add_argument("--max-substep-dt", type=float, default=0.001)
    parser.add_argument("--max-substeps", type=int, default=10)

    # Accel measurement filter (a = dv/dt)
    parser.add_argument("--accel-tau-s", type=float, default=0.3)
    parser.add_argument(
        "--teleport-dist-m",
        type=float,
        default=250.0,
        help="Teleport/reset when any vehicle travels more than this distance since last teleport.",
    )
    parser.add_argument(
        "--max-segment-restarts",
        type=int,
        default=3,
        help="Max restarts of a segment if teleport triggers mid-segment.",
    )

    # Debug dashcam (for visual inspection; can slow down and may require a desktop session)
    parser.add_argument("--dashcam", action="store_true", help="Show a dashcam window for one particle vehicle.")
    parser.add_argument("--cam-particle", type=int, default=0, help="Particle index [0..7] to attach dashcam to.")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--cam-fov", type=float, default=90.0)

    # Operating point preconditioning (NOT part of cost)
    parser.add_argument("--init-speed-mps", type=float, default=10.0, help="Target speed before starting accel schedule.")
    parser.add_argument("--init-speed-tol-mps", type=float, default=0.25, help="Tolerance to consider speed reached.")
    parser.add_argument("--init-timeout-s", type=float, default=15.0, help="Max time to reach init speed.")
    parser.add_argument("--init-hold-s", type=float, default=0.5, help="Hold at init speed for this long before schedule.")
    parser.add_argument("--init-kp", type=float, default=0.12, help="P gain for preconditioning speed controller.")
    parser.add_argument("--init-u0", type=float, default=0.18, help="Base throttle bias for preconditioning.")

    # Desired accel schedule (testbench)
    parser.add_argument("--a-pos", type=float, default=1.0)
    parser.add_argument("--a-neg", type=float, default=-1.0)
    parser.add_argument("--hold-s", type=float, default=4.0)
    parser.add_argument("--rest-s", type=float, default=2.0)
    parser.add_argument("--cycles", type=int, default=2)

    # Mode hysteresis (effort u in [-1,1])
    parser.add_argument("--deadband-on", type=float, default=0.03)
    parser.add_argument("--deadband-off", type=float, default=0.01)

    # Evaluation horizon
    parser.add_argument("--trial-s", type=float, default=40.0)
    parser.add_argument("--print-every-s", type=float, default=2.0)

    # Abort/guards per particle
    parser.add_argument("--abort-speed-mps", type=float, default=70.0)
    parser.add_argument("--abort-min-speed-mps", type=float, default=0.5)
    parser.add_argument("--abort-stuck-s", type=float, default=2.0, help="Abort if speed stays below abort-min-speed-mps for this long.")
    parser.add_argument("--min-mode-samples", type=int, default=200, help="Require both +a and -a segments to be represented.")

    # PSO settings (fixed particle count=8)
    parser.add_argument("--max-it", type=int, default=30)
    parser.add_argument("--w", type=float, default=0.7)
    parser.add_argument("--c1", type=float, default=1.5)
    parser.add_argument("--c2", type=float, default=1.5)

    # Parameter bounds: [kp, ki, kd, N]
    parser.add_argument("--kp-min", type=float, default=0.0)
    parser.add_argument("--kp-max", type=float, default=0.8)
    parser.add_argument("--ki-min", type=float, default=0.0)
    parser.add_argument("--ki-max", type=float, default=0.1)
    parser.add_argument("--kd-min", type=float, default=0.0)
    parser.add_argument("--kd-max", type=float, default=0.05)
    parser.add_argument("--n-min", type=float, default=5.0)
    parser.add_argument("--n-max", type=float, default=20.0)

    # Cost weights (inner PID)
    parser.add_argument("--w-a-pos", type=float, default=1.0, help="avg|a_err| during +a segments")
    parser.add_argument("--w-a-neg", type=float, default=1.0, help="avg|a_err| during -a segments")
    parser.add_argument("--w-u", type=float, default=0.05, help="avg|u|")
    parser.add_argument("--w-du", type=float, default=0.05, help="avg|du|")
    parser.add_argument("--w-sat", type=float, default=0.5, help="saturation ratio penalty")

    parser.add_argument("-o", type=str, default="pso_innerpid_carla_log.json")
    args = parser.parse_args()

    rng = random.Random(int(args.seed))

    # Hard-coded swarm size
    swarm_n = 8
    spawn_indices = [int(x.strip()) for x in str(args.spawn_indices).split(",") if x.strip()]
    if len(spawn_indices) != swarm_n:
        raise ValueError(f"--spawn-indices must contain exactly {swarm_n} indices, got {len(spawn_indices)}")

    schedule = _build_accel_schedule(args)
    schedule_total = sum(d for d, _ in schedule)
    if float(args.trial_s) < float(schedule_total):
        raise ValueError(f"--trial-s ({args.trial_s}) must be >= accel schedule total ({schedule_total})")

    Ts = float(args.ts)
    beta = _clip(Ts / max(1e-6, float(args.accel_tau_s)), 0.0, 1.0)

    bounds_min = [float(args.kp_min), float(args.ki_min), float(args.kd_min), float(args.n_min)]
    bounds_max = [float(args.kp_max), float(args.ki_max), float(args.kd_max), float(args.n_max)]

    def clamp_x(x: list[float]) -> list[float]:
        return [
            _clip(x[0], bounds_min[0], bounds_max[0]),
            _clip(x[1], bounds_min[1], bounds_max[1]),
            _clip(x[2], bounds_min[2], bounds_max[2]),
            _clip(x[3], bounds_min[3], bounds_max[3]),
        ]

    # Initialize swarm
    swarm: list[ParticleState] = []
    for _ in range(swarm_n):
        x = [rng.uniform(bounds_min[i], bounds_max[i]) for i in range(4)]
        v = [rng.uniform(-1.0, 1.0) for _ in range(4)]
        swarm.append(
            ParticleState(
                x=x,
                v=v,
                pbest_x=list(x),
                pbest_J=float("inf"),
                mode="coast",
                prev_u=0.0,
                last_speed=0.0,
                accel_filt=0.0,
                aborted=False,
                abort_reason="",
                sum_abs_aerr_pos=0.0,
                sum_abs_aerr_neg=0.0,
                n_pos=0,
                n_neg=0,
                sum_abs_u=0.0,
                sum_abs_du=0.0,
                sat_hits=0,
                samples=0,
            )
        )

    gbest_x = list(swarm[0].x)
    gbest_J = float("inf")

    logs = []

    import carla  # type: ignore

    client = carla.Client(args.host, int(args.port))
    client.set_timeout(10.0)
    client.load_world(str(args.map))
    time.sleep(1.0)
    world = client.get_world()
    carla_map = world.get_map()
    bp_library = world.get_blueprint_library()

    orig_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = Ts
    # If we want camera images, do not run no_rendering_mode (camera may be blank otherwise).
    settings.no_rendering_mode = bool(args.no_rendering) and (not bool(args.dashcam))
    settings.substepping = bool(args.substepping)
    settings.max_substep_delta_time = float(args.max_substep_dt)
    settings.max_substeps = int(args.max_substeps)
    world.apply_settings(settings)

    spawn_points = carla_map.get_spawn_points()
    vehicles = _spawn_vehicles(world, bp_library, spawn_points, spawn_indices, str(args.blueprint))

    dashcam_stop = threading.Event()
    dashcam_thread = None
    cam_sensor = None
    img_queue: queue.Queue | None = None

    def _dashcam_worker() -> None:
        if not _HAVE_CV2 or img_queue is None:
            return
        try:
            cv2.namedWindow("PSO dashcam", cv2.WINDOW_NORMAL)
        except Exception:
            return
        try:
            while not dashcam_stop.is_set():
                try:
                    payload = img_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if payload is None:
                    continue
                raw_bytes, width, height = payload
                try:
                    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                    arr = np.reshape(arr, (height, width, 4))
                    arr = arr[:, :, :3]
                    cv2.imshow("PSO dashcam", arr)
                    cv2.waitKey(1)
                except Exception:
                    pass
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _maybe_start_dashcam() -> None:
        nonlocal dashcam_thread, cam_sensor, img_queue
        if not bool(args.dashcam):
            return
        if not _HAVE_CV2:
            print("Dashcam requested but OpenCV/numpy are not available; continuing without dashcam.")
            return
        cam_idx = int(args.cam_particle)
        if not (0 <= cam_idx < len(vehicles)):
            raise ValueError("--cam-particle must be in [0..7]")

        try:
            img_queue = queue.Queue(maxsize=1)
            cam_bp = bp_library.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(int(args.cam_width)))
            cam_bp.set_attribute("image_size_y", str(int(args.cam_height)))
            cam_bp.set_attribute("fov", str(float(args.cam_fov)))

            # Simple chase-style dashcam attached to the vehicle.
            cam_tf = carla.Transform(
                carla.Location(x=-6.0, y=0.0, z=2.5),
                carla.Rotation(pitch=-10.0, yaw=0.0, roll=0.0),
            )
            cam_sensor = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicles[cam_idx])

            def _cb(img) -> None:
                if img_queue is None:
                    return
                payload = (bytes(img.raw_data), int(img.width), int(img.height))
                try:
                    while True:
                        img_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    img_queue.put_nowait(payload)
                except queue.Full:
                    pass

            cam_sensor.listen(_cb)
            dashcam_thread = threading.Thread(target=_dashcam_worker, name="pso_dashcam", daemon=True)
            dashcam_thread.start()
            print(f"Dashcam ON (particle {cam_idx})")
        except Exception as e:
            print(f"Failed to start dashcam: {e}")

    def cleanup():
        nonlocal cam_sensor, dashcam_thread
        try:
            dashcam_stop.set()
        except Exception:
            pass
        if img_queue is not None:
            try:
                img_queue.put_nowait(None)
            except Exception:
                pass
        if dashcam_thread is not None:
            try:
                dashcam_thread.join(timeout=1.0)
            except Exception:
                pass
        if cam_sensor is not None:
            try:
                cam_sensor.stop()
            except Exception:
                pass
            try:
                cam_sensor.destroy()
            except Exception:
                pass
        try:
            for v in vehicles:
                try:
                    v.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                except Exception:
                    pass
        except Exception:
            pass
        try:
            world.apply_settings(orig_settings)
        except Exception:
            pass
        try:
            for v in vehicles:
                try:
                    v.destroy()
                except Exception:
                    pass
        except Exception:
            pass

    def teleport_reset() -> None:
        # Reset all to spawn transforms and force stop.
        for i, idx in enumerate(spawn_indices):
            tf = spawn_points[int(idx)]
            # keep duplicate-offset consistency
            used_count = 0
            for j in range(i):
                if spawn_indices[j] == idx:
                    used_count += 1
            tf2 = carla.Transform(tf.location, tf.rotation)
            if used_count > 0:
                fwd = tf2.get_forward_vector()
                tf2.location = tf2.location + fwd * float(10.0 * used_count)
            tf2.location.z = float(tf.location.z) - 0.3
            vehicles[i].set_transform(tf2)
            # IMPORTANT: do not use set_target_velocity(0) here; it can prevent the vehicle from accelerating
            # during preconditioning (depending on CARLA version / API semantics).
            vehicles[i].apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))

    last_locs = [None for _ in range(swarm_n)]
    dist_since_tp = [0.0 for _ in range(swarm_n)]

    def reset_distance_tracking() -> None:
        for i in range(swarm_n):
            try:
                last_locs[i] = vehicles[i].get_transform().location
            except Exception:
                last_locs[i] = None
            dist_since_tp[i] = 0.0

    def update_distance_tracking() -> float:
        """Update per-vehicle traveled distance; returns max distance since last teleport."""
        max_d = 0.0
        for i in range(swarm_n):
            try:
                loc = vehicles[i].get_transform().location
            except Exception:
                continue
            prev = last_locs[i]
            if prev is not None:
                dist_since_tp[i] += math.dist((loc.x, loc.y, loc.z), (prev.x, prev.y, prev.z))
            last_locs[i] = loc
            max_d = max(max_d, float(dist_since_tp[i]))
        return float(max_d)

    def precondition_to_speed() -> list[bool]:
        """Bring each vehicle close to init-speed using a simple P controller (not tuned).

        Returns a per-vehicle reached flag.
        """
        target = float(args.init_speed_mps)
        tol = float(args.init_speed_tol_mps)
        timeout_s = float(args.init_timeout_s)
        hold_s = float(args.init_hold_s)
        kp = float(args.init_kp)
        u0 = float(args.init_u0)

        reached = [False for _ in range(swarm_n)]
        hold_left = [hold_s for _ in range(swarm_n)]

        max_steps = int(round(timeout_s / Ts))
        for _ in range(max_steps):
            world.tick()
            max_d = update_distance_tracking()
            if float(args.teleport_dist_m) > 0.0 and max_d >= float(args.teleport_dist_m):
                teleport_reset()
                for _k in range(3):
                    world.tick()
                reset_distance_tracking()
            all_done = True
            for i in range(swarm_n):
                vel = vehicles[i].get_velocity()
                speed = _planar_speed(vel)
                err = float(target - speed)

                if reached[i]:
                    # Keep holding for a short window, then coast.
                    hold_left[i] -= Ts
                    if hold_left[i] <= 0.0:
                        vehicles[i].apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
                        continue
                    # Light holding controller during hold window
                    u = _clip(u0 + kp * err, -1.0, 1.0)
                else:
                    u = _clip(u0 + kp * err, -1.0, 1.0)
                    if abs(err) <= tol and speed > 1.0:
                        reached[i] = True

                if u >= 0.0:
                    vehicles[i].apply_control(carla.VehicleControl(throttle=float(u), brake=0.0, steer=0.0))
                else:
                    vehicles[i].apply_control(carla.VehicleControl(throttle=0.0, brake=float(-u), steer=0.0))

                if not reached[i] or hold_left[i] > 0.0:
                    all_done = False

            if all_done:
                break
        return reached

    try:
        # initial stabilization tick
        _maybe_start_dashcam()
        teleport_reset()
        for _ in range(10):
            world.tick()
        reset_distance_tracking()

        print(f"Spawned {len(vehicles)} vehicles at {spawn_indices} on {args.map}.")
        print(f"PSO: particles={swarm_n} max_it={args.max_it} Ts={Ts}")

        kb_aw = 1.0

        for it in range(0, int(args.max_it) + 1):
            # Evaluate swarm (all particles simultaneously) over a fixed horizon.
            for p in swarm:
                p.reset_runtime()

            # One stateful PID instance per particle (reset at trial start).
            pids = []
            for p in swarm:
                kp, ki, kd, N = p.x
                pid = PID(
                    float(kp),
                    float(ki),
                    float(kd),
                    float(N),
                    Ts,
                    -1.0,
                    1.0,
                    kb_aw,
                    der_on_meas=True,
                    integral_disc_method="backeuler",
                    derivative_disc_method="tustin",
                )
                pid.reset()
                pids.append(pid)

            teleport_reset()
            for _ in range(3):
                world.tick()
            reset_distance_tracking()
            # reset speed histories after teleport
            for i, p in enumerate(swarm):
                v0 = _planar_speed(vehicles[i].get_velocity())
                p.last_speed = float(v0)
                p.accel_filt = 0.0

            # For each schedule segment, recondition back to init-speed, then apply the accel step.
            # This avoids the vehicle reaching very low speeds during braking segments.
            next_print = float(args.print_every_s)
            t_s = 0.0
            for seg_idx, (seg_dur_s, seg_a_sp) in enumerate(schedule):
                # Precondition all vehicles to the operating speed BEFORE each step.
                reached = precondition_to_speed()
                for i, ok in enumerate(reached):
                    if not ok and not swarm[i].aborted:
                        swarm[i].aborted = True
                        swarm[i].abort_reason = "precondition_failed"
                        vehicles[i].apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))

                # Reset differentiator and PID state so each segment starts cleanly.
                for i, p in enumerate(swarm):
                    v0 = _planar_speed(vehicles[i].get_velocity())
                    p.last_speed = float(v0)
                    p.accel_filt = 0.0
                    p.prev_u = 0.0
                    p.mode = "coast"
                    pids[i].reset()

                # Run this segment; if the distance teleport triggers mid-segment, restart it.
                restarts = 0
                while True:
                    if restarts > int(args.max_segment_restarts):
                        for i, p in enumerate(swarm):
                            if not p.aborted:
                                p.aborted = True
                                p.abort_reason = "segment_restart_limit"
                        break

                    a_sp = float(seg_a_sp)
                    low_speed_time = [0.0 for _ in range(swarm_n)]
                    segment_active = [True for _ in range(swarm_n)]
                    n_steps_seg = int(round(float(seg_dur_s) / Ts))

                    teleport_triggered = False
                    for _ in range(n_steps_seg):
                        t_s += Ts
                        world.tick()
                        max_d = update_distance_tracking()
                        if float(args.teleport_dist_m) > 0.0 and max_d >= float(args.teleport_dist_m):
                            teleport_triggered = True
                            break

                        for i, p in enumerate(swarm):
                            if p.aborted:
                                vehicles[i].apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
                                continue
                            if not segment_active[i]:
                                vehicles[i].apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
                                continue

                            vel = vehicles[i].get_velocity()
                            speed = _planar_speed(vel)
                            raw_accel = (float(speed) - float(p.last_speed)) / Ts
                            p.last_speed = float(speed)
                            p.accel_filt = _lowpass(float(p.accel_filt), float(raw_accel), beta)
                            a_meas = float(p.accel_filt)

                            u_effort = float(pids[i].step(float(a_sp), float(a_meas)))
                            u_effort = _clip(u_effort, -1.0, 1.0)

                            p.mode = _mode_from_effort(
                                p.mode,
                                float(u_effort),
                                float(args.deadband_on),
                                float(args.deadband_off),
                            )

                            throttle_cmd = _clip(u_effort, 0.0, 1.0) if p.mode == "throttle" else 0.0
                            brake_cmd = _clip(-u_effort, 0.0, 1.0) if p.mode == "brake" else 0.0

                            if throttle_cmd >= 0.999 or brake_cmd >= 0.999:
                                p.sat_hits += 1

                            vehicles[i].apply_control(
                                carla.VehicleControl(throttle=float(throttle_cmd), brake=float(brake_cmd), steer=0.0)
                            )

                            u = float(throttle_cmd) - float(brake_cmd)
                            du = float(u - float(p.prev_u))
                            p.prev_u = float(u)

                            a_err = float(a_sp - a_meas)
                            if float(a_sp) > 0.05:
                                p.sum_abs_aerr_pos += abs(float(a_err))
                                p.n_pos += 1
                            elif float(a_sp) < -0.05:
                                p.sum_abs_aerr_neg += abs(float(a_err))
                                p.n_neg += 1

                            p.sum_abs_u += abs(float(u))
                            p.sum_abs_du += abs(float(du))
                            p.samples += 1

                            if float(speed) > float(args.abort_speed_mps):
                                p.aborted = True
                                p.abort_reason = f"speed>{args.abort_speed_mps}"
                                continue

                            if float(speed) < float(args.abort_min_speed_mps):
                                if float(a_sp) < -0.05:
                                    segment_active[i] = False
                                    vehicles[i].apply_control(
                                        carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
                                    )
                                    continue
                                low_speed_time[i] += Ts
                                if low_speed_time[i] >= float(args.abort_stuck_s):
                                    p.aborted = True
                                    p.abort_reason = f"stuck<{args.abort_min_speed_mps}mps"
                                    continue
                            else:
                                low_speed_time[i] = 0.0

                        if t_s >= next_print:
                            p0 = swarm[0]
                            v0 = _planar_speed(vehicles[0].get_velocity())
                            print(
                                f"it={it:02d} t={t_s:6.2f}s seg={seg_idx+1}/{len(schedule)} a_sp={a_sp:6.2f} "
                                f"v0={v0:5.2f} a0={p0.accel_filt:6.2f} mode0={p0.mode:7s} "
                                f"dist_max={max(dist_since_tp):.1f}m"
                            )
                            next_print += float(args.print_every_s)

                    if not teleport_triggered:
                        break

                    # Restart this segment after a distance-based teleport.
                    restarts += 1
                    teleport_reset()
                    for _ in range(3):
                        world.tick()
                    reset_distance_tracking()
                    reached = precondition_to_speed()
                    for i, ok in enumerate(reached):
                        if not ok and not swarm[i].aborted:
                            swarm[i].aborted = True
                            swarm[i].abort_reason = "precondition_failed"
                            vehicles[i].apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
                    for i, p in enumerate(swarm):
                        v0 = _planar_speed(vehicles[i].get_velocity())
                        p.last_speed = float(v0)
                        p.accel_filt = 0.0
                        p.prev_u = 0.0
                        p.mode = "coast"
                        pids[i].reset()

            # Compute costs and update pbest/gbest
            J_values = []
            for i, p in enumerate(swarm):
                sat_ratio = float(p.sat_hits / max(1, p.samples))

                if p.n_pos < int(args.min_mode_samples) or p.n_neg < int(args.min_mode_samples):
                    p.aborted = True
                    if not p.abort_reason:
                        p.abort_reason = f"insufficient mode samples (pos={p.n_pos}, neg={p.n_neg})"

                avg_aerr_pos = float(p.sum_abs_aerr_pos / max(1, p.n_pos))
                avg_aerr_neg = float(p.sum_abs_aerr_neg / max(1, p.n_neg))
                avg_u = float(p.sum_abs_u / max(1, p.samples))
                avg_du = float(p.sum_abs_du / max(1, p.samples))

                J = float(
                    float(args.w_a_pos) * avg_aerr_pos
                    + float(args.w_a_neg) * avg_aerr_neg
                    + float(args.w_u) * avg_u
                    + float(args.w_du) * avg_du
                    + float(args.w_sat) * sat_ratio
                )
                if p.aborted:
                    J = float(J + 1e6)

                J_values.append(float(J))

                # Update pbest
                if float(J) < float(p.pbest_J):
                    p.pbest_J = float(J)
                    p.pbest_x = list(p.x)

                # Update gbest
                if float(J) < float(gbest_J):
                    gbest_J = float(J)
                    gbest_x = list(p.x)

                logs.append(
                    {
                        "iter": int(it),
                        "particle": int(i),
                        "J": float(J),
                        "metrics": {
                            "avg_aerr_pos": avg_aerr_pos,
                            "avg_aerr_neg": avg_aerr_neg,
                            "avg_u": avg_u,
                            "avg_du": avg_du,
                            "sat_ratio": sat_ratio,
                            "n_pos": int(p.n_pos),
                            "n_neg": int(p.n_neg),
                            "samples": int(p.samples),
                            "aborted": bool(p.aborted),
                            "abort_reason": str(p.abort_reason),
                        },
                        "x": [float(v) for v in p.x],
                        "pbest_x": [float(v) for v in p.pbest_x],
                        "pbest_J": float(p.pbest_J),
                        "gbest_x": [float(v) for v in gbest_x],
                        "gbest_J": float(gbest_J),
                    }
                )

            print(
                f"Iteration {it} best J={gbest_J:.6f} gbest_x={gbest_x} "
                f"mean J={sum(J_values)/len(J_values):.4f}"
            )
            if all(p.aborted for p in swarm):
                reasons = [p.abort_reason for p in swarm]
                print(f"  All particles aborted. Reasons: {reasons}")

            # Stop after final evaluation
            if it >= int(args.max_it):
                break

            # PSO update: velocities and positions
            for p in swarm:
                for d in range(4):
                    r1 = rng.random()
                    r2 = rng.random()
                    p.v[d] = (
                        float(args.w) * float(p.v[d])
                        + float(args.c1) * float(r1) * (float(p.pbest_x[d]) - float(p.x[d]))
                        + float(args.c2) * float(r2) * (float(gbest_x[d]) - float(p.x[d]))
                    )
                    p.x[d] = float(p.x[d]) + float(p.v[d])
                p.x = clamp_x(p.x)

    finally:
        cleanup()
        # Always save what we have so far (even partial logs).
        try:
            with open(str(args.o), "w") as f:
                json.dump(logs, f, indent=2)
            print(f"Wrote log to {args.o}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
