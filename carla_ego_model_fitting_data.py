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

            try:
                raw_bytes, width, height = payload[:3]
                array = np.frombuffer(raw_bytes, dtype=np.uint8)
                array = np.reshape(array, (height, width, 4))
                array = array[:, :, :3]  # drop alpha

                cv2.imshow("Dashboard camera", array)
                if cv2.waitKey(1) == ord('q'):
                    break
            except KeyboardInterrupt:
                print("Keyboard Interrupt caught!")
                print("Cleaning up OpenCV...")
                cv2.destroyAllWindows()
                print("Destroyed all windows")

    finally:
        print("Exiting dashcam")
        print("Cleaning up OpenCV...")
        cv2.destroyAllWindows()
        print("Destroyed all windows")

#####################################
# Main CARLA process
#####################################
def main(args, img_queue, char_queue, img_lock):
    print("CARLA longitudinal test with synchronous mode")
    print("Use CTRL+C to exit")

    latest_frame = None
    frame_lock = threading.Lock()

    # CARLA Client setup
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    print("Available maps:")
    for m in client.get_available_maps():
        print("  ", m)

    # Town04 has long highways
    client.load_world('Town04')
    time.sleep(2.0)
    world = client.get_world()
    bp_library = world.get_blueprint_library()

    # Enable synchronous mode with dt=0.01s
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.01
    settings.no_rendering_mode = False
    settings.substepping = True
    settings.max_substep_delta_time = 0.001
    settings.max_substeps = 10
    world.apply_settings(settings)
    print("Synchronous mode ON, dt = 0.01 s")

    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[54]
    spawn_point.location.z -= 0.3

    vehicle_bp = bp_library.find('vehicle.tesla.model3')
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"Spawned {vehicle.type_id} at spawn point {spawn_point}")
    vehicle.set_autopilot(False)

    cam_bp = bp_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '480')
    cam_bp.set_attribute('fov', '90')

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

    # Shared IMU acceleration state (X axis)
    imu_accel_x = 0.0
    imu_lock = threading.Lock()

    imu_bp = bp_library.find('sensor.other.imu')
    imu_bp.set_attribute('sensor_tick', '0.01')
    imu_transform = carla.Transform()
    imu = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)
    print(f"Spawned {imu.type_id}")

    def imu_cb(imu_data):
        nonlocal imu_accel_x
        with imu_lock:
            imu_accel_x = float(imu_data.accelerometer.x)

    os.makedirs("spawn_point_imgs", exist_ok=True)

    imu.listen(imu_cb)
    cam.listen(cam_cb)

    # ---------- PRBS helpers ----------
    def prbs_bit(rng):
        return 1 if rng.integers(0, 2) == 1 else -1

    def clamp(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    # ---------- CSV logging ----------
    csv_filename_root = args.f
    csv_files = []
    csv_writers = []
    for i in range(3):
        fname = f"{csv_filename_root}_range{i+1}.csv"
        f = open(fname, mode="w", newline="")
        w = csv.writer(f)
        w.writerow(["time_s", "throttle", "brake", "speed", "accel"])
        f.flush()
        csv_files.append(f)
        csv_writers.append(w)

    # ---------- Test configuration ----------
    dt = settings.fixed_delta_seconds
    total_time = 3600.0
    total_steps = int(total_time / dt)

    print(f"Running PRBS test: dt={dt}s, total_time={total_time}s")
    print(f"Logging to: {csv_filename_root}_range[1-3].csv")

    teleport_time = int(7.0 / dt)

    # One dict per speed-range test. u0 is operating point; PRBS is around u0 and clipped.
    # Adjust these to your desired operating points/ranges.
    ranges = [
        {"u0": 0.70, "umin": 0.63, "umax": 0.75, "amp": 0.02},  # high speed range
        {"u0": 0.55, "umin": 0.45, "umax": 0.65, "amp": 0.02},  # mid speed range
        {"u0": 0.35, "umin": 0.00, "umax": 0.45, "amp": 0.02},  # low speed range
    ]

    # PRBS chip timing: either fixed or random in [chip_time_min_s, chip_time_max_s]
    use_random_chip_time = True
    chip_time_s = 1.0
    chip_time_min_s = 0.5
    chip_time_max_s = 2.5

    # Range duration (how long to log PRBS after stabilization)
    dwell_time_s = 300.0  # recommended 300–600s; increase for better identifiability
    dwell_steps = int(dwell_time_s / dt)

    finished = False

    try:
        sim_time = 0.0
        start_timestamp = None
        step = 0

        current_range = 0
        throttle_cmd = 0.0

        # Stability detection before starting PRBS logging (speed must be "steady")
        band = 0.2
        stable_time_s = 4.0
        stable_steps_required = int(stable_time_s / dt)
        stable_steps = 0
        last_speed_for_stability = 0.0
        in_test_phase = False

        # PRBS state
        rng = np.random.default_rng(12345)  # reproducible PRBS
        chip_steps = max(1, int(chip_time_s / dt))
        chip_counter = 0
        prbs_state = prbs_bit(rng)

        # Range timing
        dwell_counter = 0

        # Vehicle reset
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        while not finished:
            world.tick()
            snapshot = world.get_snapshot()
            t_now = snapshot.timestamp.elapsed_seconds
            if start_timestamp is None:
                start_timestamp = t_now
            sim_time = t_now - start_timestamp

            if sim_time >= total_time:
                finished = True

            # Teleport every few seconds to stay in flat segment
            if step > 0 and step % teleport_time == 0:
                vehicle.set_transform(spawn_point)
                print("Teleport!!")

            vel = vehicle.get_velocity()
            speed = math.sqrt(vel.x**2 + vel.y**2)  # m/s

            # Use actor acceleration (you can swap to IMU if preferred)
            accel = vehicle.get_acceleration().x

            brake_cmd = 0.0

            # Done all ranges?
            if current_range >= len(ranges):
                finished = True

            if not finished:
                cfg = ranges[current_range]
                base_throttle = float(cfg["u0"])
                u_min = float(cfg["umin"])
                u_max = float(cfg["umax"])
                amp = float(cfg["amp"])

                if not in_test_phase:
                    # Approach phase: hold base_throttle until speed stabilizes
                    throttle_cmd = base_throttle

                    if abs(speed - last_speed_for_stability) <= band:
                        stable_steps += 1
                    else:
                        last_speed_for_stability = speed
                        stable_steps = 0

                    if stable_steps >= stable_steps_required:
                        in_test_phase = True
                        dwell_counter = 0

                        chip_counter = 0
                        prbs_state = prbs_bit(rng)
                        if use_random_chip_time:
                            chip_time_s = float(rng.uniform(chip_time_min_s, chip_time_max_s))
                            chip_steps = max(1, int(chip_time_s / dt))

                        print(
                            f"Range {current_range+1}: speed stabilized at "
                            f"{speed*3.6:.1f} km/h, starting PRBS test phase "
                            f"(u0={base_throttle:.3f}, amp=±{amp:.3f}, clamp=[{u_min:.3f},{u_max:.3f}])."
                        )
                else:
                    # PRBS around base throttle, clipped
                    u_candidate = base_throttle + amp * prbs_state
                    throttle_cmd = clamp(u_candidate, u_min, u_max)

                    # Update PRBS chip
                    chip_counter += 1
                    if chip_counter >= chip_steps:
                        chip_counter = 0
                        prbs_state = prbs_bit(rng)

                        if use_random_chip_time:
                            chip_time_s = float(rng.uniform(chip_time_min_s, chip_time_max_s))
                            chip_steps = max(1, int(chip_time_s / dt))

                    # End this range after dwell_time_s
                    dwell_counter += 1
                    if dwell_counter >= dwell_steps:
                        current_range += 1
                        in_test_phase = False
                        stable_steps = 0
                        last_speed_for_stability = speed
                        print(
                            f"Completed PRBS test for range {current_range} "
                            f"at t={sim_time:.2f}s, speed={speed*3.6:.1f} km/h"
                        )

            # Apply control
            control = carla.VehicleControl(
                throttle=float(throttle_cmd),
                brake=float(brake_cmd),
                steer=0.0,
                hand_brake=False,
                reverse=False
            )
            vehicle.apply_control(control)

            # Log only during test phase for current range
            if (not finished) and in_test_phase and current_range < len(csv_writers):
                writer = csv_writers[current_range]
                writer.writerow([
                    f"{sim_time:.4f}",
                    f"{throttle_cmd:.4f}",
                    f"{brake_cmd:.4f}",
                    f"{speed:.4f}",
                    f"{accel:.4f}",
                ])
                csv_files[current_range].flush()

            if step % 50 == 0:
                print(
                    f"t={sim_time:7.2f}s | range={current_range+1 if not finished else 'done'} "
                    f"| phase={'PRBS' if in_test_phase else 'approach'} "
                    f"| u={throttle_cmd:.3f} | v={speed:.2f} m/s"
                )

            step += 1

        print("PRBS test finished successfully.")
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        time.sleep(1.0)

    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopping test early.")

    finally:
        print("Cleaning up actors...")
        try:
            cam.stop()
        except:
            pass
        try:
            imu.stop()
        except:
            pass
        try:
            cam.destroy()
        except:
            pass
        try:
            imu.destroy()
        except:
            pass
        try:
            vehicle.destroy()
        except:
            pass

        for f in csv_files:
            try:
                f.close()
            except Exception:
                pass

        # Restore world settings
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        print("Done cleanup.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA longitudinal identification test (PRBS)")
    parser.add_argument('--host', type=str, help="CARLA host", default='localhost')
    parser.add_argument('--port', type=int, help="CARLA port", default=2000)
    parser.add_argument('-f', type=str, help="Output csv filename", default="out.csv")
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
