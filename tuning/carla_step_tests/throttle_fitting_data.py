"""
CARLA data collection for throttle-to-acceleration mapping.

What it does
  - Drives to operating points and applies throttle perturbations.
  - Logs speed and acceleration for model fitting.

Outputs
  - CSV logs per setpoint.

Run (example)
  python3 tuning/carla_step_tests/throttle_fitting_data.py -f throttle_out.csv
"""

import carla
import numpy as np
import cv2
import argparse
import queue
import multiprocessing
import time
import threading
import math
import os
import csv
from lib.PID import PID


#####################################
# Dashcam process
#####################################
def dashcam(args, img_queue, char_queue, img_lock):
    if args.no_cam:
        return
    time.sleep(3)
    cv2.namedWindow("Dashboard camera", cv2.WINDOW_NORMAL)
    payload = None
    try:
        while True:
            try:
                payload = img_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if payload is None:
                continue

            try:
                raw_bytes, width, height = payload[:3]
                array = np.frombuffer(raw_bytes, dtype=np.uint8)
                array = np.reshape(array, (height, width, 4))
                array = array[:, :, :3]  # drop alpha

                cv2.imshow("Dashboard camera", array)
                if cv2.waitKey(1) == ord("q"):
                    break
            except KeyboardInterrupt:
                break
    finally:
        cv2.destroyAllWindows()


def _safe_spawn_point(spawn_points, index):
    if not spawn_points:
        raise RuntimeError("No spawn points found in map.")
    if 0 <= index < len(spawn_points):
        return spawn_points[index]
    print(f"Warning: spawn point {index} out of range (0..{len(spawn_points)-1}), using 0.")
    return spawn_points[0]


def _try_spawn_vehicle(world, vehicle_bp, spawn_tf, max_tries=25, dz_per_try=0.2):
    tf = carla.Transform(
        location=carla.Location(
            x=spawn_tf.location.x,
            y=spawn_tf.location.y,
            z=spawn_tf.location.z,
        ),
        rotation=spawn_tf.rotation,
    )
    for _ in range(max_tries):
        veh = world.try_spawn_actor(vehicle_bp, tf)
        if veh is not None:
            return veh, tf
        tf.location.z += dz_per_try
    raise RuntimeError(
        f"Failed to spawn vehicle after {max_tries} tries starting at {spawn_tf.location}."
    )


def _closest_spawn_point(spawn_points, loc: carla.Location) -> carla.Transform:
    best_sp = None
    best_dist2 = float("inf")
    for sp in spawn_points:
        dx = sp.location.x - loc.x
        dy = sp.location.y - loc.y
        dz = sp.location.z - loc.z
        d2 = dx * dx + dy * dy + dz * dz
        if d2 < best_dist2:
            best_dist2 = d2
            best_sp = sp
    return best_sp


def lane_keep_steer(
    vehicle: carla.Vehicle,
    carla_map: carla.Map,
    lookahead_m: float = 12.0,
    steer_gain: float = 1.0,
    wheelbase_m: float = 2.8,
) -> float:
    tf = vehicle.get_transform()
    wp = carla_map.get_waypoint(
        tf.location, project_to_road=True, lane_type=carla.LaneType.Driving
    )
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
    return float(np.clip(steer_gain * steer_angle, -1.0, 1.0))


#####################################
# Main CARLA process
#####################################
def main(args, img_queue, char_queue, img_lock):
    print("CARLA longitudinal throttle identification test (synchronous mode)")
    print("Use CTRL+C to exit")

    latest_frame = None
    frame_lock = threading.Lock()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    client.load_world("Town04")
    time.sleep(2.0)
    world = client.get_world()
    carla_map = world.get_map()
    bp_library = world.get_blueprint_library()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.01
    settings.no_rendering_mode = False
    settings.substepping = True
    settings.max_substep_delta_time = 0.001
    settings.max_substeps = 10
    world.apply_settings(settings)
    print(f"Synchronous mode ON, dt = {settings.fixed_delta_seconds:.3f} s")

    spawn_points = world.get_map().get_spawn_points()
    base_spawn_point = _safe_spawn_point(spawn_points, args.spawn_point)
    spawn_point = carla.Transform(
        location=carla.Location(
            x=base_spawn_point.location.x,
            y=base_spawn_point.location.y + float(args.spawn_y_offset),
            z=base_spawn_point.location.z + float(args.spawn_z_offset),
        ),
        rotation=base_spawn_point.rotation,
    )

    vehicle_bp = bp_library.find("vehicle.tesla.model3")
    vehicle, used_tf = _try_spawn_vehicle(world, vehicle_bp, spawn_point)
    print(f"Spawned {vehicle.type_id} near spawn point {args.spawn_point} at {used_tf}")
    vehicle.set_autopilot(False)

    cam = None
    if not args.no_cam:
        cam_bp = bp_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "640")
        cam_bp.set_attribute("image_size_y", "480")
        cam_bp.set_attribute("fov", "90")
        cam_transform = carla.Transform(carla.Location(x=1.2, y=0.0, z=1.2))
        cam = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        print(f"Spawned {cam.type_id}")

        def cam_cb(img):
            frame = (bytes(img.raw_data), img.width, img.height, img.frame)
            with frame_lock:
                nonlocal latest_frame
                latest_frame = frame

            try:
                while True:
                    img_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                img_queue.put_nowait(frame)
            except queue.Full:
                pass

        cam.listen(cam_cb)

    dt = settings.fixed_delta_seconds

    speed_operation_points = [
        5.5555555,  # ~20 km/h
        16.666666,  # ~60 km/h
        27.777777,  # ~100 km/h
    ]

    os.makedirs(args.out_dir, exist_ok=True)

    # PID gains (same as accel_fitting_data.py / brake_fitting_data.py)
    kp = 2.59397071e-01
    ki = 1.27381733e-01
    kd = 6.21160744e-03
    N_d = 5.0
    kb_aw = 1.0

    # Stability detection (for "operating point")
    speed_band = args.speed_band
    stable_time_s = args.stable_time_s
    stable_steps_required = int(stable_time_s / dt)

    # Throttle pulse configuration
    pulse_time_s = args.pulse_time_s
    pulse_steps = int(pulse_time_s / dt)
    num_perturbations = args.num_perturbations
    throttle_delta_mag = float(args.throttle_delta)

    # Teleport management
    teleport_dist_m = float(args.teleport_dist)
    teleport_time_steps = int(args.teleport_interval_s / dt) if args.teleport_interval_s > 0.0 else 0
    teleport_cooldown_steps_required = int(args.teleport_cooldown_s / dt)
    teleport_cooldown_steps = 0
    last_vehicle_loc = None
    dist_since_tp_m = 0.0
    pending_teleport = False
    teleport_between_tests = not bool(getattr(args, "no_teleport_between_tests", False))

    lane_keep_enabled = not bool(getattr(args, "no_lane_keep", False))
    lane_lookahead_m = float(getattr(args, "lane_lookahead", 12.0))
    lane_gain = float(getattr(args, "lane_gain", 1.0))

    # Acceleration estimation from speed
    accel_filt = 0.0
    accel_window_steps = 10
    speed_history = []
    accel_tau_s = 0.3
    accel_beta = dt / accel_tau_s

    def get_speed_mps():
        vel = vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2)

    def tick_and_update_accel():
        nonlocal accel_filt, teleport_cooldown_steps
        world.tick()

        if teleport_cooldown_steps > 0:
            teleport_cooldown_steps -= 1

        speed = get_speed_mps()
        speed_history.append(speed)
        if len(speed_history) > (accel_window_steps + 1):
            speed_history.pop(0)

        if len(speed_history) > accel_window_steps:
            v_old = speed_history[0]
            raw_accel = (speed - v_old) / (accel_window_steps * dt)
        else:
            raw_accel = 0.0

        if teleport_cooldown_steps == 0:
            accel_filt = accel_filt + accel_beta * (raw_accel - accel_filt)

        snapshot = world.get_snapshot()
        return snapshot.timestamp.elapsed_seconds, speed, accel_filt

    def teleport_now_preserve_speed(vel_to_preserve=None):
        nonlocal teleport_cooldown_steps, last_vehicle_loc, dist_since_tp_m
        vel = vel_to_preserve if vel_to_preserve is not None else vehicle.get_velocity()
        sp = _closest_spawn_point(spawn_points, spawn_point.location)
        new_loc = carla.Location(
            x=spawn_point.location.x,
            y=spawn_point.location.y,
            z=(sp.location.z - 0.3) if sp is not None else spawn_point.location.z,
        )
        vehicle.set_transform(carla.Transform(location=new_loc, rotation=spawn_point.rotation))
        vehicle.set_target_velocity(vel)
        vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        teleport_cooldown_steps = teleport_cooldown_steps_required
        last_vehicle_loc = vehicle.get_transform().location
        dist_since_tp_m = 0.0

    def teleport_if_needed(step, speed, vel, allow_teleport=True):
        nonlocal teleport_cooldown_steps, last_vehicle_loc, dist_since_tp_m, pending_teleport

        if teleport_dist_m > 0.0:
            loc = vehicle.get_transform().location
            if last_vehicle_loc is None:
                last_vehicle_loc = loc
            else:
                dist_since_tp_m += math.dist(
                    (loc.x, loc.y, loc.z),
                    (last_vehicle_loc.x, last_vehicle_loc.y, last_vehicle_loc.z),
                )
                last_vehicle_loc = loc

            if dist_since_tp_m >= teleport_dist_m:
                if not allow_teleport:
                    pending_teleport = True
                    return
                traveled_m = dist_since_tp_m
                pending_teleport = False
                teleport_now_preserve_speed()
                print(
                    f"Teleport triggered: traveled {traveled_m:.2f} m "
                    f"(threshold {teleport_dist_m:.2f} m)"
                )
                return

        if (teleport_dist_m <= 0.0) and (teleport_time_steps > 0) and (step > 0):
            if step % teleport_time_steps == 0:
                if not allow_teleport:
                    pending_teleport = True
                    return
                pending_teleport = False
                teleport_now_preserve_speed()
                print(f"Teleport!! at step={step}, speed={speed*3.6:.1f} km/h")

    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

    step = 0
    start_timestamp = None
    total_pulses = len(speed_operation_points) * int(num_perturbations)
    done_pulses = 0

    try:
        for sp_mps in speed_operation_points:
            sp_kmh = int(round(sp_mps * 3.6))
            print(f"=== Speed setpoint {sp_kmh} km/h ===")

            out_name = f"{args.froot}_{sp_kmh}kmh.csv"
            out_path = os.path.join(args.out_dir, out_name)
            print(f"Writing {out_path}")

            pid = PID(kp, ki, kd, N_d, dt, 0.0, 1.0, kb_aw, True, False)
            stable_steps = 0
            ref_throttle = 0.0

            # Approach and stabilize at operating speed
            while stable_steps < stable_steps_required:
                t_abs, speed, accel_x = tick_and_update_accel()
                if start_timestamp is None:
                    start_timestamp = t_abs
                sim_time = t_abs - start_timestamp

                vel = vehicle.get_velocity()
                teleport_if_needed(step, speed, vel, allow_teleport=True)

                steer_cmd = 0.0
                if lane_keep_enabled:
                    steer_cmd = lane_keep_steer(
                        vehicle,
                        carla_map,
                        lookahead_m=lane_lookahead_m,
                        steer_gain=lane_gain,
                    )

                throttle_cmd = float(pid.step(sp_mps, speed))
                vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=throttle_cmd,
                        brake=0.0,
                        steer=float(steer_cmd),
                    )
                )

                if abs(speed - sp_mps) <= speed_band:
                    stable_steps += 1
                else:
                    stable_steps = 0

                ref_throttle = max(0.0, min(1.0, throttle_cmd))
                if step % 50 == 0:
                    print(
                        f"[APPROACH] sp={sp_kmh} km/h, speed={speed*3.6:.1f} km/h, "
                        f"ref_throttle={ref_throttle:.3f}, stable={stable_steps}/{stable_steps_required}"
                    )
                step += 1

            print(f"Stabilized at {sp_kmh} km/h with ref_throttle={ref_throttle:.3f}")

            with open(out_path, mode="w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "time_s",
                        "speed_setpoint_mps",
                        "ref_throttle",
                        "throttle",
                        "throttle_delta",
                        "brake",
                        "speed",
                        "accel_x",
                        "pulse_index",
                    ]
                )

                for pulse_idx in range(num_perturbations):
                    done_pulses += 1
                    print(
                        f"[GLOBAL {done_pulses}/{total_pulses}] "
                        f"sp={sp_kmh} km/h, pulse={pulse_idx+1}/{num_perturbations}"
                    )

                    # Re-stabilize before each pulse
                    pid = PID(kp, ki, kd, N_d, dt, 0.0, 1.0, kb_aw, True, False)
                    stable_steps = 0
                    while stable_steps < stable_steps_required:
                        t_abs, speed, accel_x = tick_and_update_accel()
                        sim_time = t_abs - start_timestamp

                        vel = vehicle.get_velocity()
                        teleport_if_needed(step, speed, vel, allow_teleport=True)

                        steer_cmd = 0.0
                        if lane_keep_enabled:
                            steer_cmd = lane_keep_steer(
                                vehicle,
                                carla_map,
                                lookahead_m=lane_lookahead_m,
                                steer_gain=lane_gain,
                            )

                        throttle_cmd = float(pid.step(sp_mps, speed))
                        vehicle.apply_control(
                            carla.VehicleControl(
                                throttle=throttle_cmd,
                                brake=0.0,
                                steer=float(steer_cmd),
                            )
                        )

                        if abs(speed - sp_mps) <= speed_band:
                            stable_steps += 1
                        else:
                            stable_steps = 0

                        ref_throttle = max(0.0, min(1.0, throttle_cmd))
                        step += 1

                    # Pulse: choose random throttle around ref_throttle
                    throttle_cmd = float(
                        np.random.uniform(ref_throttle - throttle_delta_mag, ref_throttle + throttle_delta_mag)
                    )
                    throttle_cmd = max(0.0, min(1.0, throttle_cmd))
                    throttle_delta = throttle_cmd - ref_throttle
                    vel_before_pulse = vehicle.get_velocity()

                    for _ in range(pulse_steps):
                        t_abs, speed, accel_x = tick_and_update_accel()
                        sim_time = t_abs - start_timestamp

                        vel = vehicle.get_velocity()
                        teleport_if_needed(step, speed, vel, allow_teleport=False)

                        steer_cmd = 0.0
                        if lane_keep_enabled:
                            steer_cmd = lane_keep_steer(
                                vehicle,
                                carla_map,
                                lookahead_m=lane_lookahead_m,
                                steer_gain=lane_gain,
                            )

                        vehicle.apply_control(
                            carla.VehicleControl(
                                throttle=throttle_cmd,
                                brake=0.0,
                                steer=float(steer_cmd),
                            )
                        )

                        if teleport_cooldown_steps == 0:
                            w.writerow(
                                [
                                    f"{sim_time:.4f}",
                                    f"{sp_mps:.4f}",
                                    f"{ref_throttle:.4f}",
                                    f"{throttle_cmd:.4f}",
                                    f"{throttle_delta:.4f}",
                                    "0.0000",
                                    f"{speed:.4f}",
                                    f"{accel_x:.4f}",
                                    f"{pulse_idx}",
                                ]
                            )
                        step += 1

                    f.flush()
                    print(
                        f"[PULSE {pulse_idx+1}/{num_perturbations}] "
                        f"ref={ref_throttle:.3f}, throttle={throttle_cmd:.3f}, delta={throttle_delta:+.3f}"
                    )

                    if teleport_between_tests or pending_teleport:
                        pending_teleport = False
                        teleport_now_preserve_speed(vel_before_pulse)

        print("All throttle tests completed.")

    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopping test early.")

    finally:
        print("Cleaning up actors...")
        try:
            if cam is not None:
                cam.stop()
        except Exception:
            pass
        try:
            if cam is not None:
                cam.destroy()
        except Exception:
            pass
        try:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            time.sleep(0.5)
        except Exception:
            pass
        try:
            vehicle.destroy()
        except Exception:
            pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass
        print("Done cleanup.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA throttle identification test")
    parser.add_argument("--host", type=str, default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument(
        "--spawn-point",
        type=int,
        default=125,
        help="Town04 spawn point index",
    )
    parser.add_argument(
        "--spawn-y-offset",
        type=float,
        default=-60.0,
        help="Offset applied to spawn transform Y (matches following_ros_test.py usage)",
    )
    parser.add_argument(
        "--spawn-z-offset",
        type=float,
        default=0.0,
        help="Offset applied to spawn transform Z",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="throttle_test_csvs",
        help="Folder for output CSV files",
    )
    parser.add_argument(
        "--froot",
        type=str,
        default="throttle_delta",
        help="Output filename root (prefix)",
    )
    parser.add_argument("--num-perturbations", type=int, default=10)
    parser.add_argument("--pulse-time-s", type=float, default=2.0)
    parser.add_argument(
        "--throttle-delta",
        type=float,
        default=0.2,
        help="Random throttle delta magnitude around ref_throttle (uniform in [ref-d, ref+d])",
    )
    parser.add_argument("--stable-time-s", type=float, default=4.0)
    parser.add_argument("--speed-band", type=float, default=0.2)
    parser.add_argument(
        "--no-teleport-between-tests",
        action="store_true",
        help="Disable teleporting back to the spawn transform between throttle pulses.",
    )
    parser.add_argument(
        "--teleport-dist",
        type=float,
        default=320.0,
        help="Distance-based teleport threshold [m]; set <= 0 to disable.",
    )
    parser.add_argument(
        "--teleport-interval-s",
        type=float,
        default=0.0,
        help="Legacy time-based teleport interval [s] (used only if --teleport-dist <= 0).",
    )
    parser.add_argument("--teleport-cooldown-s", type=float, default=2.0)
    parser.add_argument(
        "--no-lane-keep",
        action="store_true",
        help="Disable lane-keeping steering (pure pursuit).",
    )
    parser.add_argument(
        "--lane-lookahead",
        type=float,
        default=12.0,
        help="Lane-keeping lookahead distance in meters.",
    )
    parser.add_argument(
        "--lane-gain",
        type=float,
        default=1.0,
        help="Lane-keeping steering gain.",
    )
    parser.add_argument("--no-cam", action="store_true", help="Disable camera display process")
    args = parser.parse_args()

    img_lock = multiprocessing.Lock()
    char_queue = multiprocessing.Queue()
    img_queue = multiprocessing.Queue(maxsize=1)

    processes = [multiprocessing.Process(target=main, args=(args, img_queue, char_queue, img_lock))]
    if not args.no_cam:
        processes.append(
            multiprocessing.Process(target=dashcam, args=(args, img_queue, char_queue, img_lock))
        )

    for p in processes:
        p.start()
    for p in processes:
        p.join()
