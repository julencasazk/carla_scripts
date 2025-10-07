import carla
import numpy as np
import sys
import argparse
import time
import signal
import queue
import cv2
import multiprocessing






def show_camera_feed(img_queue, lock):
    
    time.sleep(10)
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    try:
        while True:
            payload = None
            # Acquire lock briefly to read from the queue without blocking other process
            try:
                with lock:
                    payload = img_queue.get_nowait()
            except queue.Empty:
                # nothing to read right now
                time.sleep(0.01)
                continue

            if payload is None:
                break
            # payload is (raw_bytes, width, height)
            raw_bytes, width, height = payload
            array = np.frombuffer(raw_bytes, dtype=np.uint8)
            array = np.reshape(array, (height, width, 4))
            array = array[:, :, :3]  # Drop alpha channel
            cv2.imshow("Camera", array)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()

def main(img_queue, lock):
    

    parser = argparse.ArgumentParser(description="Print vehicle data from CARLA")
    parser.add_argument('--host', type=str, default='localhost', help='CARLA host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA port')
    args = parser.parse_args()
    
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    
    vehicle_bp = np.random.choice(blueprint_library.filter('vehicle'))
    vehicle = world.spawn_actor(vehicle_bp, np.random.choice(world.get_map().get_spawn_points()))
    vehicle.set_autopilot(True)
    print(f"Spawned {vehicle.type_id}")
    
    # calculate vehicle length and width
    vehicle_length = vehicle.bounding_box.extent.x * 2
    vehicle_width = vehicle.bounding_box.extent.y * 2
    
        
    def clean_exit(signum, frame):
        print("Exiting...")
        print("\n" * 10)  # Move cursor down to avoid overwriting
        if vehicle is not None:
            vehicle.destroy()
        sys.exit(0)


    # spawn camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '480')
    camera_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    print(f"Spawned {camera.type_id}")

    gps_sensor = None
    if not gps_sensor:
        gps_bp = blueprint_library.find('sensor.other.gnss')
        gps_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
        gps_sensor = world.spawn_actor(gps_bp, gps_transform, attach_to=vehicle)
        print(f"Spawned {gps_sensor.type_id}")
        time.sleep(1)  # Give sensor time to initialize
    
    imu_sensor = None
    if not imu_sensor:
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
        imu_sensor = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)
        print(f"Spawned {imu_sensor.type_id}")
        time.sleep(1)  # Give sensor time to initialize
        
    gps_queue = queue.Queue()
    gps_sensor.listen(gps_queue.put)
    
    # Use the multiprocessing queue provided to this process and send picklable data
    # Convert raw_data to bytes so it can be sent between processes
    def camera_callback(image):
        # Acquire lock only for the brief put operation to avoid race conditions
        with lock:
            img_queue.put((bytes(image.raw_data), image.width, image.height))
    camera.listen(camera_callback)

    imu_queue = queue.Queue()
    imu_sensor.listen(imu_queue.put)
    
    print("Waiting for a bit...")
    time.sleep(5)  # wait for a while to let sensors start

    signal.signal(signal.SIGINT, clean_exit)
    
    print("Starting to print vehicle data...")
    
    try:
        while True:
            
            gps_data = None
            imu_data = None

            try:             
                gps_data = gps_queue.get(timeout=1.0)
                imu_data = imu_queue.get(timeout=1.0)
            except queue.Empty:
                print("Waiting for GPS/IMU data...")
                continue
            except Exception as e:
                print(f"Error occurred while getting vehicle data: {e}")
                continue

            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            angular_velocity = vehicle.get_angular_velocity()
            acceleration = vehicle.get_acceleration()
            yaw = transform.rotation.yaw
            drive_direction = vehicle.get_control().reverse
            
            # Print data in-place (overwrite previous lines)
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
            print("\033[F" * 8, end="")  # Move cursor up 8 lines to overwrite previous output
            time.sleep(0.1)
    finally:
        clean_exit(None, None)


if __name__ == "__main__":
    # Create a multiprocessing.Queue and a Lock and pass them to both processes
    img_queue = multiprocessing.Queue()
    lock = multiprocessing.Lock()
    p1 = multiprocessing.Process(target=show_camera_feed, args=(img_queue, lock))
    p2 = multiprocessing.Process(target=main, args=(img_queue, lock))
    p1.start()
    p2.start()
    p2.join()
    # When main exits, signal the show process to stop
    with lock:
        img_queue.put(None)
    p1.join()
