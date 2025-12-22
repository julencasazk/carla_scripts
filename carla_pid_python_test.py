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
from PID import PID

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

    pid_low  = PID(0.43127789, 0.43676547, 0.0, 15.0, Ts, u_min, u_max, kb_aw, der_on_meas=True)
    pid_mid  = PID(0.11675119, 0.085938,   0.0, 14.90530836, Ts, u_min, u_max, kb_aw, der_on_meas=True)
    pid_high = PID(0.13408096, 0.07281374,  0.0, 12.16810135, Ts, u_min, u_max, kb_aw, der_on_meas=True)

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
        "range_id"
    ])
    csv_file.flush()

    setpoints = [22.22, 33.33, 27.78, 24.4, 30.0, 33.33, 22.22]
    idx = 0

    v1 = 11.11
    v2 = 22.22
    h  = 1.0

    current_range = "low"
    range_id_map = {"low": 0, "mid": 1, "high": 2}

    teleport_time = int(15.0 / Ts)
    band = 0.2
    min_wait_steps = int(5.0 / Ts)

    last_change_step = 0
    last_change_speed = 0.0

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
            speed = math.sqrt(vel.x**2 + vel.y**2)

            if step > 0 and step % teleport_time == 0:
                vehicle.set_transform(spawn_point)
                print("Teleport!!")

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

            if current_range != prev_range:
                print(f"Range change: {prev_range} -> {current_range} at {speed:.2f} m/s")
                if current_range == "low":
                    pid_low.reset()
                elif current_range == "mid":
                    pid_mid.reset()
                else:
                    pid_high.reset()

            if current_range == "high":
                u = pid_high.step(setpoints[idx], speed)
            elif current_range == "mid":
                u = pid_mid.step(setpoints[idx], speed)
            else:
                u = pid_low.step(setpoints[idx], speed)

            if u > 0.0:
                throttle_cmd = u if u > 0.06 else 0.0
                brake_cmd = 0.0
            else:
                throttle_cmd = 0.0
                brake_cmd = abs(u)

            if (step - last_change_step) >= min_wait_steps:
                if abs(speed - last_change_speed) <= band:
                    if idx == len(setpoints) - 1:
                        finished = True
                    idx = (idx + 1) % len(setpoints)
                    print(f"Setpoint change to {setpoints[idx]:.3f}")
                last_change_step = step
                last_change_speed = speed

            print(f"Throttle: {throttle_cmd:.3f}")
            print(f"Brake: {brake_cmd:.3f}")
            print(f"Speed: {speed:.3f}")
            print(f"Range: {current_range} (id={range_id_map[current_range]})")
            print("-" * 40)

            vehicle.apply_control(carla.VehicleControl(
                throttle=float(throttle_cmd),
                brake=float(brake_cmd),
                steer=0.0
            ))

            csv_writer.writerow([
                f"{sim_time:.4f}",
                f"{throttle_cmd:.4f}",
                f"{setpoints[idx]:.4f}",
                f"{brake_cmd:.4f}",
                f"{speed:.4f}",
                f"{accel.x:.4f}",
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
