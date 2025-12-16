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

    # CSV File for logging params
    csv_filename = args.f 
    csv_file = open(csv_filename, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    # Header 
    csv_writer.writerow([
        "time_s",
        "throttle",
        "brake",
        "speed",
        "accel"
    ])
    csv_file.flush()

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
    print(f"Logging to: {csv_filename}")

    teleport_time = int(5.0 / dt)  # interval in steps
    # For doing controlled tests (no random throttle)
    # throttle_vals = [1.0, 0.0, 0.75, 0.0, 0.5, 0.0, 0.25, 0.0]
    throttle_vals = [0.0,
                     0.8,
                     0.0,
                     0.5,
                     0.4,
                     0.7,
                     0.75,
                     0.2,
                     0.0,
                     0.1,
                     0.45,
                     0.3,
                     0.8,
                     0.0]
    idx = 0
    finished = False
    
    try:
        # Sim time counter 
        sim_time = 0.0
        start_timestamp = None
        step = 0
        # Wait the car to stabilize before doing a setpoint
        stable_step = 0        
        prev_speed = 0.0

        last_change_step = 0
        last_change_speed = 0.0
        throttle_cmd = 0.0
        band = 0.2                      # stable band value of +- <band> m/s
                                        # speed is stable if inside band for <min_wait_steps> steps 
        min_wait_steps = int(2.0 / dt)  # secons between stability checks 

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

            # Check if enough time has passed since last change
            if (step - last_change_step) >= min_wait_steps:
                # Has speed settled near last_change_speed?
                if abs(speed - last_change_speed) <= band:
                    # throttle_cmd = float(np.random.rand())
                    throttle_cmd = throttle_vals[idx]
                    if idx == (len(throttle_vals) - 1):
                        finished = True
                    idx = (idx+1) % len(throttle_vals)
                    print(f"Throttle change to {throttle_cmd:.3f} at t={sim_time:.2f}s "
                          f"(speed: d{speed:.2f} m/s)")
                last_change_step = step
                last_change_speed = speed

            print(f"Current throttle: {throttle_cmd:.3f}")

            control = carla.VehicleControl(
                throttle=float(throttle_cmd),
                brake=float(brake_cmd),
                steer=0.0,
                hand_brake=False,
                reverse=False
            )
            vehicle.apply_control(control)

            csv_writer.writerow([
                f"{sim_time:.4f}",
                f"{throttle_cmd:.4f}",
                f"{brake_cmd:.4f}",
                f"{speed:.4f}",
                f"{accel:.4f}"
            ])
            csv_file.flush()
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

        csv_file.close()

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
