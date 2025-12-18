
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
import control as ctl
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

    print("Setting up PID controller")

    Ts = 0.01
    u_min = 0.0
    u_max = 1.0
    kb_aw = 1.0
    # Low speed range PID
    kp_low =   5.61726331e-02
    ki_low =   7.85334963e-03
    kd_low =   0.0
    N_d_low =  15.0
    # Mid speed range PID
    kp_mid =   5.61726331e-02
    ki_mid =   7.85334963e-03
    kd_mid =   0.0
    N_d_mid =  15.0
    # High speed range PID
    kp_high =   5.61726331e-02
    ki_high =   7.85334963e-03
    kd_high =   0.0
    N_d_high =  15.0
    
    pid_low = PID(kp_low, ki_low, kd_low, N_d_low, Ts, u_min, u_max, kb_aw, der_on_meas=True)
    pid_mid = PID(kp_mid, ki_mid, kd_mid, N_d_mid, Ts, u_min, u_max, kb_aw, der_on_meas=True)
    pid_high = PID(kp_high, ki_high, kd_high, N_d_high, Ts, u_min, u_max, kb_aw, der_on_meas=True)


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
    settings.fixed_delta_seconds = Ts # I don't know if it's important right now, 
                                        # but should match the sampling
                                        # rate of the PID on the STM32
    settings.no_rendering_mode = False  # Camera feed is prefered, although not needed
    world.apply_settings(settings)
    print("Synchronous mode ON, dt = {Ts} s")

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

    os.makedirs("spawn_point_imgs", exist_ok=True)

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

    # CSV File for logging vehicle speed to throttle
    csv_filename = args.f 
    csv_file = open(csv_filename, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    # Header 
    csv_writer.writerow([
        "time_s",
        "throttle",
        "setpoint",
        "brake",
        "speed_m_s",
        "accel_x_m_s2",
        "accel_y_m_s2",
        "accel_z_m_s2"
    ])
    csv_file.flush()

    # Step test definition
    dt = settings.fixed_delta_seconds
    total_time = 3600.0 # Very long time, enough to input all throttle setpoints 
    total_steps = int(total_time / dt)
    print(f"Running step test: dt={dt}s, total_time={total_time}s")
    print(f"Logging to: {csv_filename}")

    teleport_time = int(15.0 / dt)  # interval in steps
    # For doing controlled tests (no random throttle)
    # Incremental tests: sp1 -> 0.0 -> sp2 -> 0.0 ...
    #setpoints = [0.0, 11.11, 5.55, 2.25, 7.5, 11.11, 0.0] # Low range
    #setpoints = [11.11, 22.22, 16.66666, 13.0, 20.0, 22.22, 11.11] # Middle range
    setpoints = [22.22, 33.33, 27.77777, 24.4, 30.0, 33.33, 22.22] # Hight range
    #setpoints = [5.55, 11.1111, 16.66666, 22.2222, 27.777777, 33.333333] # Middlepoint reference checking.
    y = np.zeros_like(total_steps)
    u = np.zeros_like(total_steps)
    
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
        min_wait_steps = int(5.0 / dt)  # secons between stability checks 

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
            accel = vehicle.get_acceleration()
            # Teleport every 5 seconds to stay in flat segment 
            # of highway continuously
            if step > 0 and step % teleport_time == 0:
                vehicle.set_transform(spawn_point)
                print("Teleport!!")
            
            speed = math.sqrt(vel.x**2 + vel.y**2)
           
            if (speed >= 22.22):
                u = pid_high.step(setpoints[idx], speed)
            elif (speed >= 11.11):
                u = pid_mid.step(setpoints[idx], speed)
            else:
                u = pid_low.step(setpoints[idx], speed)
            
            if u > 0.0:
                throttle_cmd = u if u > 0.06 else 0.0 # Deadband to avoid oscillating above 0m/s
                brake_cmd = 0.0
            else:
                throttle_cmd = 0.0
                brake_cmd = abs(u)

            # Check if enough time has passed since last change
            if (step - last_change_step) >= min_wait_steps:
                # Has speed settled near last_change_speed?
                if abs(speed - last_change_speed) <= band:
                    # throttle_cmd = float(np.random.rand())

                    if idx == (len(setpoints) - 1):
                        finished = True
                    idx = (idx+1) % len(setpoints)
                    print(f"Setpoint change to {setpoints[idx]:.3f} at t={sim_time:.2f}s "
                          f"(speed: d{speed:.2f} m/s)")
                last_change_step = step
                last_change_speed = speed

            loc = vehicle.get_location()
            print(f"Current throttle: {throttle_cmd:.3f}")
            print(f"Current brake: {brake_cmd:.3f}")
            print(f"Current Setpoint: {setpoints[idx]:.3f}")
            print(f"Current speed: {speed:.3f}")
            print(f"Current X: {loc.x:.3f}")
            print(f"Current Y: {loc.y:.3f}")
            print(f"Current X: {loc.z:.3f}")


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
                f"{setpoints[idx]:.4f}",
                f"{brake_cmd:.4f}",
                f"{speed:.4f}",
                f"{accel.x:.4f}",
                f"{accel.y:.4f}",
                f"{accel.z:.4f}",
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
