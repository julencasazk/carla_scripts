"""
Single-vehicle CARLA test of the gain-scheduled speed PID (Python only).

Outputs
  - CSV logs and optional dashcam window.

Run (example)
  python3 tuning/carla_step_tests/carla_pid_python_test.py
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
    time.sleep(5)
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

            raw_bytes, width, height = payload[:3]
            array = np.frombuffer(raw_bytes, dtype=np.uint8)
            array = np.reshape(array, (height, width, 4))
            array = array[:, :, :3]

            cv2.imshow("Dashboard camera", array)
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()

#####################################
# Main CARLA process
#####################################
def main(args, img_queue, char_queue, img_lock):
    print("CARLA longitudinal test with synchronous mode")
    print("Use CTRL+C to exit")

    Ts = 0.01
    u_min = 0.0
    u_max = 1.0
    kb_aw = 1.0

    # Old gains (kept for reference):
    # GAINS_LOW  = (0.43127789, 0.43676547, 0.0, 15.0)
    # GAINS_MID  = (0.11675119, 0.085938,   0.0, 14.90530836)
    # GAINS_HIGH = (0.13408096, 0.07281374, 0.0, 12.16810135)

    # Gains matching following_ros_test.py (ACTIVE) and compare_speedpid_carla_vs_plants.py:
    GAINS_LOW  = (0.1089194,  0.04906409, 0.0,  7.71822214)
    GAINS_MID  = (0.11494122, 0.0489494,  0.0,  5.0)
    GAINS_HIGH = (0.19021246, 0.15704399, 0.0, 12.70886655)

    # Single PID instance + bumpless gain switching.
    pid = PID(*GAINS_LOW, Ts, u_min, u_max, kb_aw, der_on_meas=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    client.load_world('Town04')
    time.sleep(2.0)

    world = client.get_world()
    bp_library = world.get_blueprint_library()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = Ts
    world.apply_settings(settings)

    spawn_point = world.get_map().get_spawn_points()[54]
    spawn_point.location.z -= 0.3

    vehicle = world.spawn_actor(bp_library.find('vehicle.tesla.model3'), spawn_point)
    vehicle.set_autopilot(False)

    cam_bp = bp_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '480')
    cam_bp.set_attribute('fov', '90')
    cam = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=1.2, y=0.0, z=1.2)),
        attach_to=vehicle
    )

    def cam_cb(img):
        frame = (bytes(img.raw_data), img.width, img.height, img.frame)
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

    csv_file = open(args.f, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "time_s",
        "throttle",
        "setpoint",
        "brake",
        "speed",
        "accel_x",
        "speed_raw",
        "range_id"
    ])
    csv_file.flush()

    # Old setpoints (kept for reference):
    # setpoints = [0.0, 11.11, 5.55, 2.555, 11.111, 0.0, 2.555, 0.000]

    # Setpoint schedule matching compare_speedpid_carla_vs_plants.py:
    #  - 30s at 0.0 m/s
    #  - then 15s plateaus through low/mid/high ranges (up + down steps)
    sp_schedule = [
        (30.0, 0.0),
        (15.0, 6.0),  (15.0, 10.0), (15.0, 7.0),  (15.0, 9.0),   # low
        (15.0, 14.0), (15.0, 21.0), (15.0, 16.0), (15.0, 20.0),  # mid
        (15.0, 25.0), (15.0, 32.0), (15.0, 27.0), (15.0, 31.0),  # high
    ]
    sp_ends = []
    acc_t = 0.0
    for dur_s, sp in sp_schedule:
        acc_t += float(dur_s)
        sp_ends.append((acc_t, float(sp)))

    v1 = 11.11
    v2 = 22.22
    h  = 1.0

    current_range = "low"
    range_id_map = {"low": 0, "mid": 1, "high": 2}

    # Gain-scheduling / control uses the same simple LPF as following_ros_cam.py.
    alpha = 0.4
    speed_filt = None

    # Distance-based teleport (like following_ros_cam.py), keep speed.
    teleport_dist_m = 250.0
    last_loc = None
    dist_since_tp_m = 0.0

    def _try_set_target_velocity(actor, v):
        if hasattr(actor, "set_target_velocity"):
            actor.set_target_velocity(v)
        elif hasattr(actor, "enable_constant_velocity"):
            actor.enable_constant_velocity(v)

    start_timestamp = None
    step = 0
    finished = False

    try:
        while not finished:
            world.tick()
            snapshot = world.get_snapshot()
            t = snapshot.timestamp.elapsed_seconds
            if start_timestamp is None:
                start_timestamp = t
            sim_time = t - start_timestamp

            if sim_time >= 3600.0:
                finished = True

            vel = vehicle.get_velocity()
            accel = vehicle.get_acceleration()
            speed_raw = math.sqrt(vel.x**2 + vel.y**2)
            if speed_filt is None:
                speed_filt = float(speed_raw)
            else:
                speed_filt = float(alpha) * float(speed_raw) + (1.0 - float(alpha)) * float(speed_filt)
            speed = float(speed_filt)

            if teleport_dist_m > 0.0:
                loc = vehicle.get_transform().location
                if last_loc is None:
                    last_loc = loc
                else:
                    dist_since_tp_m += math.dist(
                        (loc.x, loc.y, loc.z),
                        (last_loc.x, last_loc.y, last_loc.z),
                    )
                    last_loc = loc

                if dist_since_tp_m >= teleport_dist_m:
                    v_now = float(speed_raw)
                    vehicle.set_transform(spawn_point)
                    fwd = vehicle.get_transform().get_forward_vector()
                    _try_set_target_velocity(vehicle, carla.Vector3D(fwd.x * v_now, fwd.y * v_now, fwd.z * v_now))
                    dist_since_tp_m = 0.0
                    last_loc = vehicle.get_transform().location
                    print(f"Teleport!! dist_since_tp_m >= {teleport_dist_m:.1f} m")

            prev_range = current_range
            if current_range == "low":
                if speed > v1 + h:
                    current_range = "mid"
            elif current_range == "mid":
                if speed > v2 + h:
                    current_range = "high"
                elif speed < v1 - h:
                    current_range = "low"
            else:
                if speed < v2 - h:
                    current_range = "mid"

            pid_sp = None
            for t_end, sp in sp_ends:
                if sim_time <= t_end:
                    pid_sp = sp
                    break
            if pid_sp is None:
                finished = True
                pid_sp = float(sp_ends[-1][1])

            # Bumpless gain scheduling (avoid hard jumps on range change).
            if current_range != prev_range:
                print(f"Range change: {prev_range} -> {current_range} at {speed:.2f} m/s")
                if current_range == "low":
                    kp, ki, kd, N = GAINS_LOW
                elif current_range == "mid":
                    kp, ki, kd, N = GAINS_MID
                else:
                    kp, ki, kd, N = GAINS_HIGH

                # Previous applied output is throttle (u>=0 with u_min=0.0).
                pid.set_gains_bumpless(kp, ki, kd, N, pid_sp, speed, float(throttle_cmd))

            u = pid.step(pid_sp, speed)

            if u > 0.0:
                throttle_cmd = u
                brake_cmd = 0.0
            else:
                throttle_cmd = 0.0
                brake_cmd = abs(u)

            print(f"Throttle: {throttle_cmd:.3f}")
            print(f"Brake: {brake_cmd:.3f}")
            print(f"Speed: {speed:.3f}")
            print(f"Range: {current_range} (id={range_id_map[current_range]})")
            print(f"Setpoint: {pid_sp:.3f}")
            print("-" * 40)

            vehicle.apply_control(carla.VehicleControl(
                throttle=float(throttle_cmd),
                brake=float(brake_cmd),
                steer=0.0
            ))

            csv_writer.writerow([
                f"{sim_time:.4f}",
                f"{throttle_cmd:.4f}",
                f"{pid_sp:.4f}",
                f"{brake_cmd:.4f}",
                f"{speed:.4f}",
                f"{accel.x:.4f}",
                f"{speed_raw:.4f}",
                f"{range_id_map[current_range]}",
            ])
            csv_file.flush()

            step += 1

        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        time.sleep(1.0)

    finally:
        cam.stop()
        vehicle.destroy()
        csv_file.close()
        print("Done cleanup.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('-f', type=str, default="out.csv")
    args = parser.parse_args()

    img_lock = multiprocessing.Lock()
    char_queue = multiprocessing.Queue()
    img_queue = multiprocessing.Queue(maxsize=1)

    processes = [
        multiprocessing.Process(target=main, args=(args, img_queue, char_queue, img_lock)),
        multiprocessing.Process(target=dashcam, args=(args, img_queue, char_queue, img_lock)),
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
