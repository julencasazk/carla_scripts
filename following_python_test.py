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
from Platooning import Platoon, PlatoonMember

def lane_keep_steer(
    vehicle: carla.Vehicle,
    carla_map: carla.Map,
    lookahead_m: float = 12.0,
    steer_gain: float = 1.0,
    wheelbase_m: float = 2.8,
) -> float:
    """
    Simple lane-keeping steering using a pure-pursuit style lookahead waypoint.
    Returns normalized steer in [-1, 1].
    """
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
    steer = float(np.clip(steer_gain * steer_angle, -1.0, 1.0))
    return steer

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
    
    if args.plen < 1:
        print("Platoon must contain at least 1 member!")
        exit()
        
    print("CARLA longitudinal test with synchronous mode")
    print("Use CTRL+C to exit")

    Ts = 0.01

    # PID gains
    u_min = -1.0
    u_max = 1.0
    kb_aw = 1.0

    pid_low  = PID(0.43127789, 0.43676547, 0.0, 15.0, Ts, u_min, u_max, kb_aw, der_on_meas=True)
    pid_mid  = PID(0.11675119, 0.085938,   0.0, 14.90530836, Ts, u_min, u_max, kb_aw, der_on_meas=True)
    pid_high = PID(0.13408096, 0.07281374,  0.0, 12.16810135, Ts, u_min, u_max, kb_aw, der_on_meas=True)

    latest_frame = None
    frame_lock = threading.Lock()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    client.load_world('Town04')
    time.sleep(2.0)
    world = client.get_world()
    carla_map = world.get_map()
    bp_library = world.get_blueprint_library()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = Ts
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    print(f"Synchronous mode ON, dt = {Ts} s")

    spawn_points = world.get_map().get_spawn_points()
    
    # ------------------------
    # Build platoon members
    # ------------------------
    members = []
    vehicles = []

    base_spawn_point = spawn_points[125]
    base_spawn_point.location.y -= 60.0 # put the leader further ahead to let the followers have space to spawn

    # Distance between vehicles in spawn (just so they don't collide at start)
    initial_spacing = 6.0

    vehicle_bp = bp_library.find('vehicle.tesla.model3')

    for i in range(args.plen):
        # Leader is i == 0, followers i > 0 placed behind it
        spawn_tf = carla.Transform(
            location=carla.Location(
                x=base_spawn_point.location.x,
                y=base_spawn_point.location.y + i * initial_spacing,
                z=base_spawn_point.location.z
            ),
            rotation=base_spawn_point.rotation
        )
        veh = world.spawn_actor(vehicle_bp, spawn_tf)
        veh.set_autopilot(False)
        vehicles.append(veh)

        role = "lead" if i == 0 else "follower"
        name = "lead" if i == 0 else f"ego_{i}"

        member = PlatoonMember(
            name=name,
            role=role,
            vehicle=veh,
            pid=None,
            desired_time_headway=0.7 if (i == 1) else 0.3, # First follower mantain a safer distance to lead
            min_spacing=7.5,
            K_dist=5,
        )
        members.append(member)
        print(f"Spawned {veh.type_id} as {role} at {spawn_tf}")

    # First member is the leader, second member we treat as "ego" for camera/logging
    platoon = Platoon(members)
    lead_member = platoon.members[0]
    lead_vehicle = lead_member.vehicle

    if args.plen > 1:
        ego_member = platoon.members[-1]
        ego_vehicle = ego_member.vehicle
    else:
        ego_member = lead_member
        ego_vehicle = lead_vehicle

    # ------------------------
    # Camera attached to ego
    # ------------------------
    cam_bp = bp_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '480')
    cam_bp.set_attribute('fov', '90')

    cam_transform = carla.Transform(carla.Location(x=-10.0, y=0.0, z=10.0), carla.Rotation(yaw=-0.0, pitch=-35)) # transform positions seem to be relative when attached

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

    dt = settings.fixed_delta_seconds
    total_time = 3600.0
    total_steps = int(total_time / dt)
    teleport_dist_m = float(args.teleport_dist)
    dist_since_tp_m = 0.0
    last_lead_loc = None

    # Lead speed setpoints (m/s)
    lead_speed_setpoints = [0.0, 33.0, 0.0, 20.0, 25.0, 15.0, 10.0, 5.0, 15.0, 0.0]
    lead_sp_idx = 0

    finished = False
    try:
        sim_time = 0.0
        start_timestamp = None
        step = 0

        last_change_step = 0
        last_change_speed = 0.0
        band = 0.2
        min_wait_steps = int(5.0 / dt)

        # Set all vehicles to zero control initially
        for v in vehicles:
            v.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        while not finished:
            world.tick()
            snapshot = world.get_snapshot()
            t = snapshot.timestamp.elapsed_seconds
            if start_timestamp is None:
                start_timestamp = t
            sim_time = t - start_timestamp

            if sim_time >= total_time:
                finished = True

            # Distance-based teleport: when leader has traveled N meters, teleport whole platoon (keep relative positions)
            lead_loc = lead_vehicle.get_transform().location
            if last_lead_loc is None:
                last_lead_loc = lead_loc
            else:
                dist_since_tp_m += math.dist(
                    (lead_loc.x, lead_loc.y, lead_loc.z),
                    (last_lead_loc.x, last_lead_loc.y, last_lead_loc.z),
                )
                last_lead_loc = lead_loc

            if teleport_dist_m > 0.0 and dist_since_tp_m >= teleport_dist_m:
                print(f"Teleport triggered: leader traveled {dist_since_tp_m:.2f} m (threshold {teleport_dist_m:.2f} m)")
                lead_tf = lead_vehicle.get_transform()
                base_tf = base_spawn_point

                # Use a consistent rotation for all members (e.g. base spawn yaw)
                ref_rotation = base_tf.rotation

                for m in platoon.members:
                    tf = m.vehicle.get_transform()
                    rel = tf.location - lead_tf.location

                    # keep platoon geometry in x,y
                    target_loc = carla.Location(
                        x=base_tf.location.x + rel.x,
                        y=base_tf.location.y + rel.y,
                        z=base_tf.location.z + rel.z  # provisional z
                    )

                    # find closest spawn point to this x,y (only to get z)
                    sp = closest_spawn_point(spawn_points, target_loc)

                    # glue z to spawn point, keep x,y and a consistent yaw
                    new_loc = carla.Location(
                        x=target_loc.x,
                        y=target_loc.y,
                        z=sp.location.z + 0.1
                    )
                    new_tf = carla.Transform(
                        location=new_loc,
                        rotation=ref_rotation 
                    )

                    m.vehicle.set_transform(new_tf)

                print("Teleport with spawn snapping!!")
                dist_since_tp_m = 0.0
                last_lead_loc = lead_vehicle.get_transform().location

            # Decide global lead speed setpoint
            global_lead_speed_sp = lead_speed_setpoints[lead_sp_idx]

            # Update platoon
            states, controls = platoon.update(dt=dt, global_lead_speed_sp=global_lead_speed_sp)

            # Optional: add lane-keeping steering while keeping longitudinal (throttle/brake)
            if args.lane_keep:
                for m, ctrl in zip(platoon.members, controls):
                    ctrl.steer = lane_keep_steer(
                        m.vehicle,
                        carla_map,
                        lookahead_m=args.lane_lookahead,
                        steer_gain=args.lane_gain,
                    )
                    m.vehicle.apply_control(ctrl)

            # Unpack leader/ego info for logging (only if there is a follower)
            lead_speed, lead_accel_abs, lead_tf = states[0]
            lead_ctrl = controls[0]

            if args.plen > 1:
                ego_speed, ego_accel_abs, ego_tf = states[1]
                ego_ctrl = controls[1]

                ego_lead_dist = math.dist(
                    (ego_tf.location.x, ego_tf.location.y, ego_tf.location.z),
                    (lead_tf.location.x, lead_tf.location.y, lead_tf.location.z),
                )

                speed = ego_speed
                desired_dist = max(
                    ego_member.min_spacing + 3.0,
                    speed * ego_member.desired_time_headway + ego_member.min_spacing,
                )
            else:
                # Single vehicle case
                ego_speed = lead_speed
                ego_accel_abs = lead_accel_abs
                ego_ctrl = lead_ctrl
                ego_lead_dist = 0.0
                desired_dist = 0.0

            # Step logic to change lead SP when ego settles (only meaningful if there is a follower)
            if args.plen > 1 and (step - last_change_step) >= min_wait_steps:
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

            # Logging
            print(f"t={sim_time:.2f}s  leader_dist_since_tp={dist_since_tp_m:.2f} m")
            print(
                f"  Lead:  SP={global_lead_speed_sp:.2f} v={lead_speed:.2f} "
                f"Th={lead_ctrl.throttle:.3f} Br={lead_ctrl.brake:.3f}"
            )
            if args.plen > 1:
                print(
                    f"  Ego:   SP={ego_member.current_speed_sp:.2f} v={ego_speed:.2f} "
                    f"Th={ego_ctrl.throttle:.3f} Br={ego_ctrl.brake:.3f}"
                )
                print(f"  Dist ego->lead: {ego_lead_dist:.2f} m")

            csv_writer.writerow([
                f"{sim_time:.4f}",
                f"{ego_ctrl.throttle:.4f}",
                f"{ego_member.current_speed_sp if args.plen > 1 else 0.0:.4f}",
                f"{ego_ctrl.brake:.4f}",
                f"{ego_speed:.4f}",
                f"{ego_accel_abs:.4f}",
                f"{lead_ctrl.throttle:.4f}",
                f"{global_lead_speed_sp:.4f}",
                f"{lead_ctrl.brake:.4f}",
                f"{lead_speed:.4f}",
                f"{lead_accel_abs:.4f}",
                f"{ego_lead_dist:.4f}",
                f"{desired_dist:.4f}",
            ])
            csv_file.flush()

            print(f"Step {step}/{total_steps}")
            print(f"Elapsed time: {sim_time:.2f}/{total_time:.2f}")
            step += 1

        print("Step test finished successfully.")
        for v in vehicles:
            v.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
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
            world.apply_settings(original_settings)
        except:
            pass
        for v in vehicles:
            try:
                v.destroy()
            except:
                pass
        csv_file.close()
        print("Done cleanup.")

def closest_spawn_point(spawn_points, loc: carla.Location) -> carla.Transform:
    """Return the spawn point (Transform) whose location is closest to loc."""
    best_sp = None
    best_dist2 = float("inf")
    for sp in spawn_points:
        dx = sp.location.x - loc.x
        dy = sp.location.y - loc.y
        dz = sp.location.z - loc.z
        d2 = dx*dx + dy*dy + dz*dz
        if d2 < best_dist2:
            best_dist2 = d2
            best_sp = sp
    return best_sp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA longitudinal identification test")
    parser.add_argument('--host', type=str, help="CARLA host", default='localhost')
    parser.add_argument('--port', type=int, help="CARLA port", default=2000)
    parser.add_argument('-f', type=str, help="Output csv filename", default="out.csv")
    parser.add_argument('--plen', type=int, default=2, help="Number of platoon members, minimum one (leader)")
    parser.add_argument(
        '--teleport-dist',
        type=float,
        default=250.0,
        help="Teleport the platoon back when the leader has traveled this distance in meters; set <= 0 to disable.",
    )
    parser.add_argument(
        '--lane-keep',
        action='store_true',
        help="Enable simple lane-keeping steering (pure pursuit) while keeping longitudinal control from this script.",
    )
    parser.add_argument('--lane-lookahead', type=float, default=12.0, help="Lane-keeping lookahead distance in meters.")
    parser.add_argument('--lane-gain', type=float, default=1.0, help="Lane-keeping steering gain.")
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
