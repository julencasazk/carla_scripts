"""
CARLA experiment for distance/spacing behavior with logging.

Outputs
  - CSV logs and optional dashcam window.

Run (example)
  python3 experiments/distance_testing.py
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
import random

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
    settings.fixed_delta_seconds = 0.01
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    print("Synchronous mode ON, dt = 0.01 s")

    # Traffic Manager setup
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm.global_percentage_speed_difference(0.0)  # 0% slower than speed limit

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    # -------------------------------
    # Ego vehicle with sensors
    # -------------------------------
    # Find a free spawn point for ego (avoid collisions)
    vehicle_bp = bp_library.find('vehicle.tesla.model3')
    ego_vehicle = None
    ego_spawn = None

    for sp in spawn_points:
        sp.location.z -= 0.3
        ego = world.try_spawn_actor(vehicle_bp, sp)
        if ego is not None:
            ego_vehicle = ego
            ego_spawn = sp
            break

    if ego_vehicle is None:
        print("ERROR: Could not find free spawn point for ego vehicle.")
        return

    print(f"Spawned ego {ego_vehicle.type_id} at spawn point {ego_spawn}")
    ego_vehicle.set_autopilot(True, tm.get_port())

    cam_bp = bp_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '480')
    cam_bp.set_attribute('fov', '90')

    cam_transform = carla.Transform(carla.Location(x=1.2, y=0.0, z=1.2))
    cam = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle)
    print(f"Spawned {cam.type_id}")

    # Dashboard cam setup
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

    # LiDAR setup (attached to ego)
    lidar_bp = bp_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('rotation_frequency', '5')
    lidar_bp.set_attribute('points_per_second', '8000')
    lidar_bp.set_attribute('channels', '16')
    lidar_bp.set_attribute('upper_fov', '1')
    lidar_bp.set_attribute('lower_fov', '-1')

    lidar = world.spawn_actor(
        lidar_bp,
        carla.Transform(carla.Location(x=0.0, z=1.5)),
        attach_to=ego_vehicle
    )
    print(f"Spawned {lidar.type_id}")

    last_min_dist = {"value": None}
    lidar_lock = threading.Lock()

    def lidar_cb(pc: carla.LidarMeasurement):
        # LidarMeasurement is a 4D vector like [x, y, z, intensity]
        pts = np.frombuffer(pc.raw_data, dtype=np.float32)
        pts = np.reshape(pts, (-1, 4))

        xyz = pts[:, :3]
        x = xyz[:, 0]
        y = xyz[:, 1]

        in_front = (
            (x > 0.0) &
            (np.abs(y) < 1.0)
        )

        forward_pts = xyz[in_front]

        if forward_pts.size == 0:
            min_dist = None
        else:
            dists = np.linalg.norm(forward_pts[:, :2], axis=1)
            min_dist = float(np.min(dists))

        with lidar_lock:
            last_min_dist["value"] = min_dist

        if min_dist is None:
            print("No object detected ahead")
        else:
            print(f"Object detected ahead at {min_dist:.4f} m")

    lidar.listen(lidar_cb)

    # -------------------------------
    # Spawn NPC traffic (random)
    # -------------------------------

    vehicle_blueprints = bp_library.filter('vehicle.*')
    npc_vehicles = []

    # Remove ego spawn from list to avoid double-use
    free_spawns = [sp for sp in spawn_points if sp != ego_spawn]

    print(f"Attempting to spawn up to {args.nv} NPC vehicles...")
    for spawn_point in free_spawns:
        if len(npc_vehicles) >= args.nv:
            break

        vehicle_bp_npc = random.choice(vehicle_blueprints)

        # Optional: avoid bikes/peds etc by requiring 4 wheels
        if int(vehicle_bp_npc.get_attribute('number_of_wheels').as_int()) != 4:
            continue

        vehicle_bp_npc.set_attribute('role_name', 'autopilot')

        try:
            npc = world.try_spawn_actor(vehicle_bp_npc, spawn_point)
            if npc is not None:
                npc.set_autopilot(True, tm.get_port())
                npc_vehicles.append(npc)
                print(f"Spawned NPC {npc.type_id} at {spawn_point.location}")
        except RuntimeError:
            continue

    print(f"Spawned {len(npc_vehicles)} NPC vehicles.")

    # -------------------------------
    # Main sim loop
    # -------------------------------
    try:
        while True:
            world.tick()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected.\nExiting...")
        print("Cleaning up actors...")

        # Stop sensors
        try:
            lidar.stop()
        except:
            pass

        # Destroy sensors
        try:
            lidar.destroy()
            print(f"{lidar.type_id} destroyed")
        except:
            print(f"Failed to destroy {lidar.type_id}")
        try:
            cam.destroy()
            print(f"{cam.type_id} destroyed")
        except:
            print(f"Failed to destroy {cam.type_id}")

        # Destroy NPC vehicles
        for v in npc_vehicles:
            try:
                v.destroy()
                print(f"{v.type_id} destroyed")
            except:
                print(f"Failed to destroy {v.type_id}")

        # Destroy ego
        try:
            ego_vehicle.destroy()
            print(f"{ego_vehicle.type_id} destroyed")
        except:
            print(f"Failed to destroy {ego_vehicle.type_id}")

        print("Cleaning finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA longitudinal identification test")
    parser.add_argument('--host', type=str, help="CARLA host", default='localhost')
    parser.add_argument('--port', type=int, help="CARLA port", default=2000)
    parser.add_argument('--nv', type=int, default=80, help="Number of NPC vehicles")
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
