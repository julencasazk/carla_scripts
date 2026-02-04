#!/usr/bin/env python3
"""
Spawns a CARLA vehicle and shows a live RGB camera feed.

Outputs
  - OpenCV window with live camera stream.

Run (example)
  python3 sensor_tests/get_cam.py --host localhost --port 2000
"""

import carla
import numpy as np
import cv2
import argparse
import time
import signal
import sys
import queue


def clean_exit(signum, frame, camera=None, vehicle=None):
    print("Exiting...")
    if camera is not None:
        camera.stop()
    if vehicle is not None:
        vehicle.destroy()
    cv2.destroyAllWindows()
    sys.exit(0)



def main():
    
    parser = argparse.ArgumentParser(description="Get camera images from CARLA")
    parser.add_argument('--host', type=str, default='localhost', help='CARLA host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA port')
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    # spawn random vehicle 
    vehicle_bp = np.random.choice(blueprint_library.filter('vehicle'))
    vehicle = world.spawn_actor(vehicle_bp, np.random.choice(world.get_map().get_spawn_points()))
    vehicle.set_autopilot(True)
    print(f"Spawned {vehicle.type_id}")
    # spawn camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '480')
    camera_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    print(f"Spawned {camera.type_id}")
    
    signal.signal(signal.SIGINT, lambda s, f: clean_exit(s, f, camera, vehicle))
    
    img_queue = queue.Queue()
    camera.listen(img_queue.put)
    
    
    print("Waiting for a bit...")
    time.sleep(5)  # wait for a while to let the camera start
    print("Starting to capture images...")
    try:
        while True:
            image = img_queue.get()
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            
            array = array[:, :, :3]
            
            # display the image
            cv2.imshow("Camera", array)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.stop()
        vehicle.destroy()
        cv2.destroyAllWindows()
        
        
if __name__ == '__main__':
    main()
