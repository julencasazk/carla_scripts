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

    Ts = 0.01

    # PID gains (from PSO)
    kp = 2.59397071e-01
    ki = 1.27381733e-01
    kd = 6.21160744e-03
    N_d = 5.0
    kb_aw = 1.0
    u_min, u_max = -1.0, 1.0

    # Two PIDs: one for ego, one for lead
    pid_ego = PID(kp, ki, kd, N_d, Ts, u_min, u_max, kb_aw, der_on_meas=True)
    pid_lead = PID(kp, ki, kd, N_d, Ts, u_min, u_max, kb_aw, der_on_meas=True)

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
    settings.fixed_delta_seconds = Ts
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    print(f"Synchronous mode ON, dt = {Ts} s")

    spawn_points = world.get_map().get_spawn_points()
    
    # Spawn Ego Vehicle
    ego_spawn_point = spawn_points[54]
    ego_spawn_point.location.z -= 0.3
    vehicle_bp = bp_library.find('vehicle.tesla.model3')
    ego_vehicle = world.spawn_actor(vehicle_bp, ego_spawn_point)
    print(f"Spawned {ego_vehicle.type_id} at spawn point {ego_spawn_point}")
    ego_vehicle.set_autopilot(False)

    # Spawn lead NPC vehicle (20m ahead along X-, as in your comment)
    lead_distance_init = 20.0
    lead_spawn_point = carla.Transform(
        carla.Location(
            x=ego_spawn_point.location.x - lead_distance_init,
            y=ego_spawn_point.location.y,
            z=ego_spawn_point.location.z + 0.3
        ),
        rotation=ego_spawn_point.rotation
    )
    lead_bp = bp_library.find('vehicle.tesla.model3')
    lead_vehicle = world.spawn_actor(lead_bp, lead_spawn_point)
    print(f"Spawned {lead_vehicle.type_id} at spawn point {lead_spawn_point}")
    lead_vehicle.set_autopilot(False)

    # Camera
    cam_bp = bp_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '480')
    cam_bp.set_attribute('fov', '90')

    cam_transform = carla.Transform(carla.Location(x=1.2, y=0.0, z=1.2))
    cam = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle)
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

    # CSV for logging
    csv_filename = args.f 
    csv_file = open(csv_filename, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "time_s",
        "ego_throttle",
        "ego_setpoint_speed",
        "ego_brake",
        "ego_speed",
        "ego_accel_abs",
        "lead_throttle",
        "lead_setpoint_speed",
        "lead_brake",
        "lead_speed",
        "lead_accel_abs",
        "lead_dist",
        "desired_dist"
    ])
    csv_file.flush()

    # Simulation timing
    dt = settings.fixed_delta_seconds
    total_time = 3600.0
    total_steps = int(total_time / dt)
    print(f"Running step test: dt={dt}s, total_time={total_time}s")
    print(f"Logging to: {csv_filename}")

    teleport_time = int(5.0 / dt)

    # Lead speed setpoints (m/s) instead of plain throttles
    # You can tune/extend this sequence
    lead_speed_setpoints = [0.0, 33.0, 0.0, 20.0, 25.0, 15.0, 10.0, 5.0, 15.0, 0.0]
    
    desired_dist = 5 #  m of desired distance from lead to ego
    K_dist = 0.2
    lead_sp_idx = 0

    finished = False
    
    try:
        sim_time = 0.0
        start_timestamp = None
        step = 0

        # Stability logic for when to change the lead setpoint
        last_change_step = 0
        last_change_speed = 0.0
        band = 0.2                      # m/s band for "stable"
        min_wait_steps = int(5.0 / dt)  # wait before checking stability

        ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        lead_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        while not finished:
            world.tick()
            snapshot = world.get_snapshot()
            t = snapshot.timestamp.elapsed_seconds
            if start_timestamp is None:
                start_timestamp = t
            sim_time = t - start_timestamp

            if sim_time >= total_time:
                finished = True

            # Read states
            ego_vel = ego_vehicle.get_velocity()
            ego_accel = ego_vehicle.get_acceleration()
            ego_transform = ego_vehicle.get_transform()

            lead_vel = lead_vehicle.get_velocity()
            lead_accel = lead_vehicle.get_acceleration()
            lead_transform = lead_vehicle.get_transform()

            # Teleport every 5s while preserving relative pose
            if step > 0 and step % teleport_time == 0:
                rel_loc = lead_transform.location - ego_transform.location
                rel_rot = lead_transform.rotation

                ego_vehicle.set_transform(ego_spawn_point)

                new_lead_transform = carla.Transform(
                    location=carla.Location(
                        x=ego_spawn_point.location.x + rel_loc.x,
                        y=ego_spawn_point.location.y,
                        z=ego_spawn_point.location.z + 0.3
                    ),
                    rotation=ego_spawn_point.rotation
                )
                lead_vehicle.set_transform(new_lead_transform)
                print("Teleport!!")

            # Speeds and accelerations (magnitudes)
            ego_speed = math.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)
            ego_accel_abs = math.sqrt(ego_accel.x**2 + ego_accel.y**2 + ego_accel.z**2)

            lead_speed = math.sqrt(lead_vel.x**2 + lead_vel.y**2 + lead_vel.z**2)
            lead_accel_abs = math.sqrt(lead_accel.x**2 + lead_accel.y**2 + lead_accel.z**2)

            rel_vect = lead_transform.location - ego_transform.location
            ego_lead_dist = math.sqrt(rel_vect.x**2 + rel_vect.y**2 + rel_vect.z**2)
            
            desired_dist = ego_speed * 0.5 + 5 # Current speed * (reaction_time + braking time)
                                                # Assuming cars are cooperating, they receive braking
                                                # signal withing transmission time + some tens of ms for processing
                                                # We can assume a worst case scenenario of 500ms total
                                                # Also, distance is calculated from center of mass,
                                                # 5m is added approximating each car's length to be 5m
                                            
                                                
            desired_dist = max(8.0, desired_dist)  # If vehicle stopped, prevent distance 0.
        
            dist_err = ego_lead_dist - desired_dist
            if dist_err >= 0.0:
                dSetpoint = K_dist * dist_err
            if dist_err < 0.0:
                dSetpoint = 2*K_dist * dist_err
            
            

            # -------------------------
            # Lead vehicle PID: tracks its own speed setpoint
            # -------------------------
            lead_speed_sp = lead_speed_setpoints[lead_sp_idx]
            u_lead = pid_lead.step(lead_speed_sp, lead_speed)

            if u_lead > 0.0:
                lead_throttle_cmd = u_lead
                lead_brake_cmd = 0.0
            else:
                lead_throttle_cmd = 0.0
                lead_brake_cmd = abs(u_lead)

            # -------------------------
            # Ego vehicle PID: tracks current lead speed and distance
            # -------------------------
            ego_speed_sp = lead_speed_sp  + dSetpoint # Platooning, the entire platoon knows the setpoint
            u_ego = pid_ego.step(ego_speed_sp, ego_speed)

            if u_ego > 0.0:
                ego_throttle_cmd = u_ego
                ego_brake_cmd = 0.0
            else:
                ego_throttle_cmd = 0.0
                ego_brake_cmd = abs(u_ego)

            # -------------------------
            # Logic to change lead's setpoint when ego settles
            # -------------------------
            if (step - last_change_step) >= min_wait_steps:
                if abs(ego_speed - last_change_speed) <= band:
                    if lead_sp_idx == (len(lead_speed_setpoints) - 1):
                        finished = True
                    else:
                        lead_sp_idx = (lead_sp_idx + 1) % len(lead_speed_setpoints)
                    print(
                        f"Lead speed SP change to {lead_speed_setpoints[lead_sp_idx]:.3f} m/s "
                        f"at t={sim_time:.2f}s (ego speed: {ego_speed:.2f} m/s)"
                    )
                    last_change_step = step
                    last_change_speed = ego_speed
                else:
                    last_change_step = step
                    last_change_speed = ego_speed

            # Apply controls
            ego_control = carla.VehicleControl(
                throttle=float(ego_throttle_cmd),
                brake=float(ego_brake_cmd),
                steer=0.0,
                hand_brake=False,
                reverse=False
            )

            lead_control = carla.VehicleControl(
                throttle=float(lead_throttle_cmd),
                brake=float(lead_brake_cmd),
                hand_brake=False,
                reverse=False
            )

            ego_vehicle.apply_control(ego_control)
            lead_vehicle.apply_control(lead_control)

            print(f"t={sim_time:.2f}s")
            print(f"  Lead:  SP={lead_speed_sp:.2f}  v={lead_speed:.2f}  Th={lead_throttle_cmd:.3f} Br={lead_brake_cmd:.3f}")
            print(f"  Ego:   SP={ego_speed_sp:.2f}   v={ego_speed:.2f}   Th={ego_throttle_cmd:.3f} Br={ego_brake_cmd:.3f}")
            print(f"  Dist ego->lead: {ego_lead_dist:.2f} m")

            # CSV row
            csv_writer.writerow([
                f"{sim_time:.4f}",
                f"{ego_throttle_cmd:.4f}",
                f"{ego_speed_sp:.4f}",
                f"{ego_brake_cmd:.4f}",
                f"{ego_speed:.4f}",
                f"{ego_accel_abs:.4f}",
                f"{lead_throttle_cmd:.4f}",
                f"{lead_speed_sp:.4f}",
                f"{lead_brake_cmd:.4f}",
                f"{lead_speed:.4f}",
                f"{lead_accel_abs:.4f}",
                f"{ego_lead_dist:.4f}",
                f"{desired_dist:.4f}"
            ])
            csv_file.flush()

            print(f"Step {step}/{total_steps}")
            print(f"Elapsed time: {sim_time:.2f}/{total_time:.2f}")
            step += 1

        print("Step test finished successfully.")

        ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        lead_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
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
            ego_vehicle.destroy()
        except:
            pass

        try:
            lead_vehicle.destroy()
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
