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

    # Town04 has the longest highways strips of any default map 
    client.load_world('Town04')
    time.sleep(2.0)
    world = client.get_world()
    bp_library = world.get_blueprint_library()

    # Enable synchronous mode with dt=0.01s
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True  
    settings.fixed_delta_seconds = 0.01 # I don't know if it's important right now, 
                                        # but should match the sampling
                                        # rate of the PID on the STM32
    settings.no_rendering_mode = False  # Camera feed is prefered, although not needed
    settings.substepping = True
    settings.max_substep_delta_time = 0.001
    settings.max_substeps = 10 # Trying this to hopefully make the physics more reliable
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
    # Attach IMU at vehicle origin (sensor frame X will be longitudinal)
    imu_transform = carla.Transform()
    imu = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)
    print(f"Spawned {imu.type_id}")

    def imu_cb(imu_data):
        nonlocal imu_accel_x
        # IMU accelerometer is in sensor frame (m/s^2)
        with imu_lock:
            imu_accel_x = float(imu_data.accelerometer.x)

    os.makedirs("spawn_point_imgs", exist_ok=True)


    imu.listen(imu_cb)
    cam.listen(cam_cb)

    # CSV Files for logging: one per speed range test
    csv_filename_root = args.f
    csv_files = []
    csv_writers = []
    for i in range(3):
        fname = f"{csv_filename_root}_range{i+1}.csv"
        f = open(fname, mode="w", newline="")
        w = csv.writer(f)
        w.writerow([
            "time_s",
            "throttle",
            "brake",
            "speed",
            "accel",
        ])
        f.flush()
        csv_files.append(f)
        csv_writers.append(w)

    # Step test definition
    dt = settings.fixed_delta_seconds
    total_time = 3600.0 # Very long time, enough to input all throttle setpoints 
    total_steps = int(total_time / dt)
    '''
    min_sec = 10
    max_sec = 50
    min_step_btw_sp = int(min_sec / dt)
    max_step_btw_sp = int(max_sec / dt) 
    print(f"The test will run for {total_steps} steps...")
    prev_val = 0.0
    
    # Piecewise-constant throttle with random segment lengths in [10 s, 30 s]
    throttle_sp = np.zeros(total_steps, dtype=float)
    i = 0
    while i < total_steps:
        # Create a random throttle value 
        throttle_val = float(np.random.rand())  # [0, 1)
        # The wait between changes is proportional to step change
        delta = abs(throttle_val - prev_val)
        dwell_sec = min_sec + (max_sec - min_sec)*delta
        # How many steps the value is going to be applied to 
        seg_len = int(dwell_sec / dt) # Floor, makes sure it will not be over the max value
        # Check if close to end of steps
        end = min(i + seg_len, total_steps)
        # Apply throttle value for whole range 
        throttle_sp[i:end] = throttle_val
        print(f"Throttle {throttle_val:.3f} from t={i*dt:.2f}s to t={end*dt:.2f}s "
              f"(~{(end - i)*dt:.2f}s segment)")
        prev_val = throttle_val
        # Jump to next 0 value
        i = end
    '''
    print(f"Running step test: dt={dt}s, total_time={total_time}s")
    print(f"Logging to: {csv_filename_root}_range[1-3].csv")

    teleport_time = int(5.0 / dt)  # interval in steps
    # For doing controlled tests around approximate steady speeds.
    # Each sub-array corresponds to one speed range. The first value is
    # the base throttle that approximately produces the range's "middle"
    # speed. The remaining values are test throttles to apply once the
    # vehicle has reached a stable speed in that range.
    throttle_vals = [
        [0.7, 0.75, 0.7, 0.63, 0.75, 0.63, 0.7],
        [0.55, 0.65, 0.55, 0.45, 0.65, 0.45, 0.55],
        [0.35, 0.45, 0.35, 0.0, 0.45, 0.0, 0.35],
    ]

    finished = False
    
    try:
        # Sim time counter 
        sim_time = 0.0
        start_timestamp = None
        step = 0
        # Test sequencing across ranges
        current_range = 0
        throttle_cmd = 0.0

        # Stability detection for reaching the approximate steady speed
        band = 0.2                      # m/s band for "stable" speed
        stable_time_s = 4.0
        stable_steps_required = int(stable_time_s / dt)
        stable_steps = 0
        last_speed_for_stability = 0.0

        # Test phase timing for each throttle in the pattern
        dwell_time_s = 20.0 # Might be too long, ill see
        dwell_steps = int(dwell_time_s / dt)
        dwell_counter = 0
        pattern_index = 0
        in_test_phase = False

        # Vehicle reset 
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # Synchronous loop 
        #while sim_time < total_time:
        while not finished:
            # Update a single simulation step 
            world.tick()
            snapshot = world.get_snapshot()
            t = snapshot.timestamp.elapsed_seconds
            if start_timestamp is None:
                start_timestamp = t
            sim_time = t - start_timestamp
            
            # If on last step, do the step and finish
            if sim_time >= total_time:
                finished = True


            brake_cmd = 0.0  
            
            

            vel = vehicle.get_velocity()
            # Use IMU X-axis linear acceleration instead of actor acceleration
            # Uncomment when using IMU instead of Actor acceleration
            #with imu_lock:
            #    accel = imu_accel_x
            
            # Just checking if physics improved
            accel = vehicle.get_acceleration().x
            # end checking
            
            # Teleport every 5 seconds to stay in flat segment 
            # of highway continuously
            if step > 0 and step % teleport_time == 0:
                vehicle.set_transform(spawn_point)
                print("Teleport!!")
            
            speed = math.sqrt(vel.x**2 + vel.y**2)

            # If we've completed all ranges, stop the test
            if current_range >= len(throttle_vals):
                finished = True

            if not finished:
                base_throttle = throttle_vals[current_range][0]
                pattern = throttle_vals[current_range][1:]

                if not in_test_phase:
                    # Approach phase: hold base_throttle until speed is
                    # stable within a small band for a given time.
                    throttle_cmd = base_throttle

                    if abs(speed - last_speed_for_stability) <= band:
                        stable_steps += 1
                    else:
                        last_speed_for_stability = speed
                        stable_steps = 0

                    if stable_steps >= stable_steps_required:
                        in_test_phase = True
                        pattern_index = 0
                        dwell_counter = 0
                        print(
                            f"Range {current_range+1}: speed stabilized at "
                            f"{speed*3.6:.1f} km/h, starting test phase."
                        )
                else:
                    # Test phase: cycle through the throttle pattern for
                    # this range. Each value is held for dwell_time_s.
                    if pattern:
                        throttle_cmd = pattern[pattern_index]
                    else:
                        throttle_cmd = base_throttle

                    dwell_counter += 1
                    if dwell_counter >= dwell_steps:
                        dwell_counter = 0
                        pattern_index += 1
                        if pattern_index >= len(pattern):
                            # Finished this range; advance to next
                            current_range += 1
                            in_test_phase = False
                            stable_steps = 0
                            last_speed_for_stability = speed
                            print(
                                f"Completed test for range {current_range} at "
                                f"t={sim_time:.2f}s, speed={speed*3.6:.1f} km/h"
                            )

            print(f"Current throttle: {throttle_cmd:.3f} (range {current_range+1 if not finished else 'done'})")

            control = carla.VehicleControl(
                throttle=float(throttle_cmd),
                brake=float(brake_cmd),
                steer=0.0,
                hand_brake=False,
                reverse=False
            )
            vehicle.apply_control(control)

            # Log only during the test phase for the current range
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
            print(f"Step {step}/{total_steps}")
            print(f"Elapsed time: {sim_time}/{total_time}")

            step = step + 1
            

        print("Step test finished successfully.")

        # Stop vehicle when test over
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
            vehicle.destroy()
        except:
            pass

        for f in csv_files:
            try:
                f.close()
            except Exception:
                pass

        print("Done cleanup.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA longitudinal identification test")
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
