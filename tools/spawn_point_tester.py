#!/usr/bin/env python3
"""
Utility to list spawn points and spawn a vehicle with a dashcam view.

Outputs
  - Console output of spawn points and an optional camera window.

Run (example)
  python3 tools/spawn_point_tester.py --map Town04
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
import sys, time, multiprocessing
import termios, tty, select
from lib import kb_controller
import os  # ADDED

def dashcam(args, img_queue, char_queue, img_lock):
    time.sleep(5)
    cv2.namedWindow("Dashboard camera", cv2.WINDOW_NORMAL)
    payload = None
    try:
        while True:
            try:
                # Do not lock the queue; just read the latest available frame
                payload = img_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if payload is None:
                continue

            try:
                # Support payload with optional frame_id at index 3
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
        print("Exiting without keyboard interrupt")
        print("Cleaning up OpenCV...")
        cv2.destroyAllWindows()
        print("Destroyed all windows")

def main(args, img_queue, char_queue, img_lock):
    print("CARLA vehicle controller with STM32H753zi")
    print("Use CTRL+C to exit (it's not clean right now :/ )")

    # Keep latest frame in-process for snapshots
    latest_frame = None  # (raw_bytes, width, height, frame_id)
    frame_lock = threading.Lock()

    # CARLA Client setup
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    map = client.get_available_maps()
    print(f"Available maps:\n{map}")
    client.load_world(args.map)
    time.sleep(10)
    world = client.get_world()
    bp_library = world.get_blueprint_library()

    spawn_points = world.get_map().get_spawn_points()
    print(f"Possible spawn points on {map}:\n{spawn_points}")
    print(f"Number of possible spawn points on {map}:\n{len(spawn_points)}")
    
    vehicle_bp = bp_library.find('vehicle.tesla.model3')
    vehicle = world.spawn_actor(vehicle_bp, 
                                spawn_points[0])
    print(f"Spawned {vehicle.type_id} at first available spawn point {spawn_points[0]}")
    
    vehicle.set_autopilot(False)  # Change for normal mode
    physics_control = vehicle.get_physics_control()
    
    cam_bp = bp_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '1920')
    cam_bp.set_attribute('image_size_y', '1080')
    cam_transform = carla.Transform(carla.Location(x=1.2, y=0.0, z=1.2))
    cam = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    print(f"Spawned {cam.type_id}")
    
    # Directory for snapshots
    os.makedirs("spawn_point_imgs", exist_ok=True)

    spawn_point_idx = 0 

    def cam_cb(img):
        # Prepare frame tuple and publish to display queue; also store latest for snapshots
        frame = (bytes(img.raw_data), img.width, img.height, img.frame)
        # Update latest frame (thread-safe within this process)
        with frame_lock:
            nonlocal latest_frame
            latest_frame = frame
        # Push to display process (drop stale frames)
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
    
    try:
        while True:
            wheel_friction_fr = physics_control.wheels[0].tire_friction
            wheel_friction_fl = physics_control.wheels[1].tire_friction
            wheel_friction_rr = physics_control.wheels[2].tire_friction
            wheel_friction_rl = physics_control.wheels[3].tire_friction
            ''' 
            print(f"Wheel Friction - \nFR: {wheel_friction_fr:.2f}, \nFL: {wheel_friction_fl:.2f}, "
                  f"\nRR: {wheel_friction_rr:.2f}, \nRL: {wheel_friction_rl:.2f}")
            '''
            spawn_point_idx += 1
            if (spawn_point_idx >= len(spawn_points)):
                spawn_point_idx = 0
            
            print(f"Going to next spawn location: \n\n{spawn_points[spawn_point_idx]}\n")
            print(f"Spawn point Index: {spawn_point_idx} / {len(spawn_points)}\n")
            vehicle.set_transform(spawn_points[spawn_point_idx])

            # Wait for a new frame (by frame_id) after the teleport
            prev_id = None
            with frame_lock:
                if latest_frame is not None:
                    prev_id = latest_frame[3]

            snapshot = None
            deadline = time.time() + 2.0  # up to 2s to get a new frame at new location
            while time.time() < deadline:
                with frame_lock:
                    if latest_frame is not None:
                        if prev_id is None or latest_frame[3] != prev_id:
                            snapshot = latest_frame
                            break
                time.sleep(0.01)

            if snapshot is not None:
                raw_bytes, width, height, _ = snapshot
                array = np.frombuffer(raw_bytes, dtype=np.uint8)
                array = np.reshape(array, (height, width, 4))
                array = array[:, :, :3]  # drop alpha
                dir_path = f"{args.map}_spawn_point_imgs"
                os.makedirs(dir_path, exist_ok=True)
                meta_path = f"{dir_path}/{args.map}_spawn_points.txt"
                if not os.path.exists(meta_path):
                    open(meta_path, "w").close()
                cv2.imwrite(f"{args.map}_spawn_point_imgs/{args.map}_spawn_point_{spawn_point_idx:03d}.png", array)
                sp = spawn_points[spawn_point_idx]
                loc, rot = sp.location, sp.rotation

                with open(f"{args.map}_spawn_point_imgs/{args.map}_spawn_points.txt", "a") as f:
                    f.write(
                        f"spawn_point_{spawn_point_idx:03d}.png "
                        f"loc({loc.x:.2f},{loc.y:.2f},{loc.z:.2f}) "
                        f"rot(pitch={rot.pitch:.2f},yaw={rot.yaw:.2f},roll={rot.roll:.2f})\n"
                    )
                
            else:
                print("Warning: no new frame received for snapshot after teleport")

            time.sleep(3)
    except KeyboardInterrupt:
        cam.stop()
        vehicle.destroy()
        print("Destroyed vehicle actor")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STM32H7 Algorithm Testing Node")
    parser.add_argument('--host', type=str, help="CARLA host", default='localhost')
    parser.add_argument('--port', type=int, help="CARLA port", default=2000)
    parser.add_argument('--map', type=str, help="CARLA map to load", default='Town01')
    args = parser.parse_args()
    
    img_lock = multiprocessing.Lock()  # Not used anymore, but kept for signature compatibility
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
        
        kb = kb_controller.KBHit()
        char = 0
        while True:
            if kb.kbhit():
                c = kb.getch()
                char_queue.put(c)
                if ord(c) == 27:
                    break
    kb.set_normal_term()
