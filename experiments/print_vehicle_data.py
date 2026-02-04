"""
CARLA experiment to print vehicle physics data and show a camera feed.

Outputs
  - Console prints and an optional camera window.

Run (example)
  python3 experiments/print_vehicle_data.py
"""

import carla
import numpy as np
import sys
import argparse
import time
import signal
import queue
import cv2
import multiprocessing


def show_camera_feed(img_queue, lock, flag):
    
    time.sleep(10)
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    try:

        while True:

            if flag.is_set():
                break
            
            payload = None
            try:
                with lock:
                    payload = img_queue.get_nowait()
            except queue.Empty:
                # nothing to read right now
                time.sleep(0.01)
                continue

            if payload is None:
                break
            
            raw_bytes, width, height = payload
            array = np.frombuffer(raw_bytes, dtype=np.uint8)
            array = np.reshape(array, (height, width, 4))
            array = array[:, :, :3]  # Without alpha channel
            cv2.imshow("Camera", array)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("Cleaning up OpenCV process...")
        cv2.destroyAllWindows()
        print("Destroyed all windows...")


def main(img_queue, lock, flag):
    
    parser = argparse.ArgumentParser(description="Print vehicle data from CARLA")
    parser.add_argument('--host', type=str, default='localhost', help='CARLA host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA port')
    args = parser.parse_args()
    
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    
    # spawn a model 3 vehicle
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle = world.spawn_actor(vehicle_bp, np.random.choice(world.get_map().get_spawn_points()))
    vehicle.set_autopilot(True)
    print(f"Spawned {vehicle.type_id}")
    
    physics_control = vehicle.get_physics_control()
    max_steer_angle = physics_control.wheels[0].max_steer_angle
    print(f"Max Steering Angle: {max_steer_angle}")
    
    # calculate vehicle length and width
    vehicle_length = vehicle.bounding_box.extent.x * 2
    vehicle_width = vehicle.bounding_box.extent.y * 2
    

    # spawn camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '480')
    camera_transform = carla.Transform(carla.Location(x=0.8, y=1.2, z=1.0), carla.Rotation(yaw=15))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    print(f"Spawned {camera.type_id}")
    

    # spawn GPS
    gps_sensor = None
    if not gps_sensor:
        gps_bp = blueprint_library.find('sensor.other.gnss')
        gps_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
        gps_sensor = world.spawn_actor(gps_bp, gps_transform, attach_to=vehicle)
        print(f"Spawned {gps_sensor.type_id}")
        time.sleep(1)  # Give sensor time to initialize
    # spawn IMU  
    imu_sensor = None
    if not imu_sensor:
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
        imu_sensor = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)
        print(f"Spawned {imu_sensor.type_id}")
        time.sleep(1)  # Give sensor time to initialize
        
    gps_queue = queue.Queue()
    gps_sensor.listen(gps_queue.put)
    
    def camera_callback(image):
        with lock:
            img_queue.put((bytes(image.raw_data), image.width, image.height))

    camera.listen(camera_callback)

    imu_queue = queue.Queue()
    imu_sensor.listen(imu_queue.put)
    
    print("Waiting for a bit...")
    time.sleep(5)  # wait for a while to let sensors start


    
    print("Starting to print vehicle data...")

    try:
        while True:

            if flag.is_set():
                break

            try:
                gps_data = gps_queue.get(timeout=1.0)
                imu_data = imu_queue.get(timeout=1.0)
            except queue.Empty:
                print("Waiting for GPS/IMU data...")
                continue
            except Exception as e:
                print(f"Error occurred while getting vehicle data: {e}")
                continue
            
            gps_queue.queue.clear()
            imu_queue.queue.clear()

            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            angular_velocity = vehicle.get_angular_velocity()
            yaw_rate = angular_velocity.y * 180 / np.pi  # Convert to degrees per second
            acceleration = vehicle.get_acceleration()
            
            '''
            Steering angle depends on the wheelspan and the steering input.
            Vehicle steering input is a float in range [-1.0, 1.0] and max_steer_angle
            is the maximum steering angle of the front wheels in degrees.
            max_steer_angle is obtained from the vehicle's physics control.
            '''
            steering_angle = vehicle.get_control().steer * max_steer_angle
            yaw = transform.rotation.yaw
            # Curvature calculation 
            wheelbase = 2.875  # in m for Tesla Model 3
            '''
            I took this formula from the internet, I still don't
            understand it completely.
            '''
            turn_radius = wheelbase / np.sin(np.radians(steering_angle)) if steering_angle != 0 else float('inf')
            curvature = 1 / turn_radius if turn_radius != float('inf') else 0
            # Is the vehicle in reverse?
            drive_direction = vehicle.get_control().reverse
            
            # Print all data
            print("-" * 40)
            
            print(f"Vehicle Type: {vehicle.type_id}            ")
            print(f"  Yaw: {yaw}            ")
            print(f"  Location: {transform.location}            ")
            print(f"  Rotation: {transform.rotation}            ")
            print(f"  Velocity: {velocity}            ")
            print(f"  Angular Velocity: {angular_velocity}            ")
            print(f"  Acceleration: {acceleration}            ")
            print(f"  GPS Latitude: {gps_data.latitude}            ")
            print(f"  GPS Longitude: {gps_data.longitude}            ")
            print(f"  GPS Altitude: {gps_data.altitude}            ")
            print(f"  IMU Accelerometer: {imu_data.accelerometer}            ")
            print(f"  IMU Gyroscope: {imu_data.gyroscope}            ")
            print(f"  IMU Compass: {imu_data.compass * 180 / np.pi}            ")
            print(f"  Drive Direction (Reverse): {drive_direction}            ")
            print(f"  Vehicle Length: {vehicle_length}            ")
            print(f"  Vehicle Width: {vehicle_width}            ")
            print(f"  Steering Angle: {steering_angle}            ")
            print(f"  Angular velocity (Yaw Rate): {yaw_rate}            ")
            print(f"  Relation between steering angle and yaw rate: {steering_angle / yaw_rate if yaw_rate != 0 else 0}            ")
            print(f"  Wheelbase: {wheelbase}            ")
            print(f"  Turn Radius: {turn_radius}            ")
            print(f"  Curvature: {curvature}            ")
            
            print("-" * 40)
            print("\033[F" * 22, end="")  # Move cursor up 22 lines to overwrite previous output

    finally:
        print("Cleaning up CARLA process...")
        # Exit cleanly
        if gps_sensor is not None:
            gps_sensor.stop()
            gps_sensor.destroy()
        if imu_sensor is not None:
            imu_sensor.stop()
            imu_sensor.destroy()
        if camera is not None:
            camera.stop()
            camera.destroy()
        if vehicle is not None:
            vehicle.destroy()
        sys.exit(0)
        


if __name__ == "__main__":
    
    img_queue = multiprocessing.Queue()
    lock = multiprocessing.Lock()

    flag = multiprocessing.Event()
    processes = [
        multiprocessing.Process(target=show_camera_feed, args=(img_queue, lock, flag)),
        multiprocessing.Process(target=main, args=(img_queue, lock, flag))
    ]

    def terminate_all(signum, frame):
        print("Terminating all processes...")
        flag.set()

    signal.signal(signal.SIGINT, terminate_all)

    for p in processes:
        p.start()
    for p in processes:
        p.join()
