"""
CARLA data collection for acceleration-response identification.

What it does
  - Holds fixed speed setpoints and applies random throttle perturbations.
  - Estimates acceleration from speed derivative + low-pass filter.
  - Teleports to keep the vehicle on a safe road segment.

Outputs
  - CSV files per setpoint: time, throttle, throttle_delta, brake, speed, accel.

Run (example)
  python3 tuning/carla_step_tests/accel_fitting_data.py -f accel_out.csv
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
    #settings.substepping = True
    #settings.max_substep_delta_time = 0.001
    #settings.max_substeps = 10 # Trying this to hopefully make the physics more reliable
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

    os.makedirs("spawn_point_imgs", exist_ok=True)

    # Start camera sensor
    cam.listen(cam_cb)

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

    # Speed operation points (m/s).
    # The PID controller will get the necessary throttle
    # value to maintain the speed setpoint and, once stable,
    # we will switch to open-loop throttle perturbation tests.
    speed_operation_points = [
        5.5555555,  # ~20 km/h
        16.666666,  # ~60 km/h
        27.777777   # ~100 km/h
    ]

    # CSV files: one per speed operation point
    csv_filename_root = args.f
    csv_files = []
    csv_writers = []
    for r in speed_operation_points:
        # Use setpoint value in km/h for the filename
        sp_kmh = int(round(r * 3.6))
        fname = f"{csv_filename_root}_{sp_kmh}kmh.csv"
        f = open(fname, mode="w", newline="")
        w = csv.writer(f)
        # time, absolute throttle, throttle delta, brake, speed, accel
        w.writerow(["time_s", "throttle", "throttle_delta", "brake", "speed", "accel_x"])
        csv_files.append(f)
        csv_writers.append(w)

    print(f"Running step test: dt={dt}s, total_time={total_time}s")
    print(f"Base filename: {csv_filename_root}")

    teleport_time = int(15.0 / dt)  # interval in steps
    # After each teleport, ignore samples for this cooldown window so that
    # teleport-induced transients do not enter the logged data.
    teleport_cooldown_s = 2.0
    teleport_cooldown_steps_required = int(teleport_cooldown_s / dt)

    # PID gains
    kp = 2.59397071e-01
    ki = 1.27381733e-01
    kd = 6.21160744e-03
    N_d = 5.0
    kb_aw = 1.0

    # Control states
    STATE_APPROACH = 0   # Use PID to reach and hold setpoint
    STATE_TEST = 1       # Open-loop throttle perturbation tests around ref_throttle

    # Stability detection
    speed_band = 0.2                 # m/s band for "stable" speed
    stable_time_s = 4.0              # time inside band before declaring stable
    stable_steps_required = int(stable_time_s / dt)

    # Perturbation test configuration
    delta_time_s = 7.5               # duration of each random delta (s)
    base_time_s = 5.0                # unused now, kept for reference
    delta_steps = int(delta_time_s / dt)
    base_steps = int(base_time_s / dt)
    num_perturbations = 30           # how many random deltas per setpoint
    
    try:
        # Sim time counter 
        sim_time = 0.0
        start_timestamp = None
        step = 0

        # Index into speed_operation_points
        setpoint_idx = 0

        # Loop termination flag
        finished = False

        # Initialize PID for first setpoint
        pid = PID(kp, ki, kd, N_d, dt, 0.0, 1.0, kb_aw, True, False)

        # State machine variables
        state = STATE_APPROACH
        throttle_cmd = 0.0
        brake_cmd = 0.0
        ref_throttle = 0.0
        throttle_delta = 0.0

        # Stabilization tracking
        stable_steps = 0

        # Perturbation test tracking
        perturbation_index = 0
        phase_is_delta = False  # False => at ref_throttle, True => at ref_throttle + delta
        phase_step_counter = 0
        current_delta = 0.0

        # Teleport cooldown counter for excluding samples from logs
        teleport_cooldown_steps = 0

        # Acceleration estimation from speed: multi-step finite difference
        # plus low-pass filtering to strongly attenuate sample-to-sample
        # speed jitter.
        accel_filt = 0.0
        # Use a backward difference over this window (e.g. 0.1 s for N=10)
        accel_window_steps = 10
        speed_history = []
        # First-order low-pass time constant (s) for acceleration estimate
        accel_tau_s = 0.3
        accel_beta = dt / accel_tau_s

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


            vel = vehicle.get_velocity()
            speed = math.sqrt(vel.x**2 + vel.y**2)

            # Teleport periodically back to the original spawn point to stay
            # in a safe road segment. Preserve target velocity and start the
            # cooldown so teleport-induced artifacts are excluded from logs.
            if step > 0 and step % teleport_time == 0:
                vehicle.set_transform(spawn_point)
                vehicle.set_target_velocity(vel)
                vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))

                teleport_cooldown_steps = teleport_cooldown_steps_required
                print(
                    f"Teleport!! back to spawn at t={sim_time:.2f}s, "
                    f"speed={speed*3.6:.1f} km/h"
                )

            # Acceleration estimation: multi-step finite difference of speed
            # over a short window plus IIR low-pass. This behaves like
            # differentiating a 0.1 s averaged speed, which greatly reduces
            # sample-to-sample noise.
            speed_history.append(speed)
            if len(speed_history) > (accel_window_steps + 1):
                # Keep only the last N+1 samples
                speed_history.pop(0)

            if len(speed_history) > accel_window_steps:
                v_old = speed_history[0]
                raw_accel = (speed - v_old) / (accel_window_steps * dt)
            else:
                raw_accel = 0.0

            if teleport_cooldown_steps > 0:
                # During cooldown, keep the previous filtered acceleration and
                # just count down; these samples will not be logged.
                teleport_cooldown_steps -= 1
            else:
                accel_filt = accel_filt + accel_beta * (raw_accel - accel_filt)

            current_setpoint = speed_operation_points[setpoint_idx]

            if state == STATE_APPROACH:
                # Use PID to drive speed toward the current setpoint
                throttle_cmd = pid.step(current_setpoint, speed)
                throttle_delta = 0.0

                # Check if speed is within the stability band
                if abs(speed - current_setpoint) <= speed_band:
                    stable_steps += 1
                else:
                    stable_steps = 0

                # When stable long enough, capture reference throttle and
                # transition to open-loop test state
                if stable_steps >= stable_steps_required:
                    ref_throttle = max(0.0, min(1.0, throttle_cmd))
                    print(
                        f"Setpoint {current_setpoint:.2f} m/s stabilized at t={sim_time:.2f}s "
                        f"with ref_throttle={ref_throttle:.3f}"
                    )

                    # Reset test state counters and pick the first random delta
                    perturbation_index = 0
                    phase_is_delta = False
                    phase_step_counter = 0
                    # Slightly larger deltas for higher setpoints
                    base_delta_mag = 0.02
                    scale = 1.0 + 0.3 * float(setpoint_idx)
                    max_delta = base_delta_mag * scale
                    current_delta = float(np.random.uniform(-max_delta, max_delta))

                    state = STATE_TEST

            elif state == STATE_TEST:
                # Open-loop throttle perturbation tests around ref_throttle.
                # Apply a random delta around ref_throttle for delta_time_s,
                # then immediately switch to another random delta without
                # explicitly returning to ref_throttle in between.
                if perturbation_index < num_perturbations:
                    # Apply current random delta around ref_throttle
                    throttle_cmd = ref_throttle + current_delta
                    throttle_cmd = max(0.0, min(1.0, throttle_cmd))
                    throttle_delta = throttle_cmd - ref_throttle
                    phase_step_counter += 1

                    if phase_step_counter >= delta_steps:
                        # Move on to the next random delta
                        phase_step_counter = 0
                        perturbation_index += 1
                        if perturbation_index < num_perturbations:
                            base_delta_mag = 0.02
                            scale = 1.0 + 0.3 * float(setpoint_idx)
                            max_delta = base_delta_mag * scale
                            current_delta = float(np.random.uniform(-max_delta, max_delta))
                else:
                    # Finished perturbations for this setpoint
                    setpoint_idx += 1
                    if setpoint_idx >= len(speed_operation_points):
                        print("All speed setpoint tests completed.")
                        finished = True
                    else:
                        # Prepare for next setpoint: new PID instance, reset tracking
                        pid = PID(kp, ki, kd, N_d, dt, 0.0, 1.0, kb_aw, True, False)
                        state = STATE_APPROACH
                        stable_steps = 0
                        throttle_delta = 0.0
                        print(
                            f"Advancing to next setpoint: {speed_operation_points[setpoint_idx]:.2f} m/s"
                        )

                # Log data only during the test phase and skip samples
                # during the teleport cooldown window so that teleport-
                # induced transients are not included in the dataset.
                if (not finished) and (teleport_cooldown_steps == 0):
                    writer = csv_writers[setpoint_idx]
                    writer.writerow([
                        f"{sim_time:.4f}",
                        f"{throttle_cmd:.4f}",
                        f"{throttle_delta:.4f}",
                        f"{brake_cmd:.4f}",
                        f"{speed:.4f}",
                        f"{accel_filt:.4f}",
                    ])
                    csv_files[setpoint_idx].flush()
                

            control = carla.VehicleControl(
                throttle=float(throttle_cmd),
                brake=float(brake_cmd),
                steer=0.0,
                hand_brake=False,
                reverse=False
            )
            vehicle.apply_control(control)

            # Console debug information for the current test
            state_name = "APPROACH" if state == STATE_APPROACH else "TEST"
            print(
                f"[Setpoint {setpoint_idx+1}/{len(speed_operation_points)}] "
                f"sp={current_setpoint*3.6:.1f} km/h, "
                f"speed={speed*3.6:.1f} km/h, "
                f"accel_x={accel_filt:.2f} m/s^2, "
                f"throttle={throttle_cmd:.3f}, "
                f"delta={throttle_delta:.3f}, "
                f"test={perturbation_index}/{num_perturbations}, "
                f"state={state_name}, "
                f"t={sim_time:.2f}s, step={step}/{total_steps}"
            )

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

        # Close all CSV files
        for f in csv_files:
            try:
                f.close()
            except:
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
