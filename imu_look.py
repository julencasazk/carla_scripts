#! /usr/bin/env python3

import rclpy
import carla
import numpy as np
import cv2
import argparse
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32
from std_msgs.msg import Bool
from sensor_msgs.msg import Imu
import queue
import multiprocessing
import time
import threading
import math


# Axis Map for quickly changing axis from real IMU to CARLA.
# e.g. rolling the imu left pitches carla up: 'PITCH_FROM': ('roll', -1)
AXIS_MAP = {
    'ROLL_FROM':  ('pitch', +1),
    'PITCH_FROM': ('roll', +1), 
    'YAW_FROM':   ('yaw',  -1), 
}


def dash_cam(img_queue, img_lock):
    time.sleep(5) 
    cv2.namedWindow("Dashboard Camera", cv2.WINDOW_NORMAL)
    try:
        while True:
            payload = None
            try:
                with img_lock:
                    payload = img_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.01)
                continue 
            if payload is None:
                break   
            
            raw_bytes, width, height = payload
            array = np.frombuffer(raw_bytes, dtype=np.uint8)
            array = np.reshape(array, (height, width, 4))
            array = array[:,:,:3] # Discard alpha channel
            
            cv2.imshow("Dashboard Camera", array)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    finally:
        print("Cleaning up OpenCV...")
        cv2.destroyAllWindows()
        print("Destroyed all windows")

def quat_to_euler_deg(x, y, z, w):
    # Normalize quaternion
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n == 0.0:
        return 0.0, 0.0, 0.0
    x, y, z, w = x/n, y/n, z/n, w/n

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

class CarlaRosNode(Node):
    def __init__(self, camera, base_transform):
        super().__init__('carla_cam_control')
        qos = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT)
        self._imu_sub = self.create_subscription(Imu, 'imu_data', self.imu_cb, qos)
        self._camera = camera
        self._base_loc = base_transform.location
        self._base_rot = base_transform.rotation
        self._rotation_mult = 3
        
    # Replace axis between them following the map
    def _apply_axis_map(self, roll, pitch, yaw):
        vals = {'roll': roll, 'pitch': pitch, 'yaw': yaw}
        def pick(which, sign):
            return sign * vals[which]
        r = pick(*AXIS_MAP['ROLL_FROM'])
        p = pick(*AXIS_MAP['PITCH_FROM'])
        y = pick(*AXIS_MAP['YAW_FROM'])
        return r, p, y 


    def imu_cb(self, msg: Imu):
        # IMU quaternion -> Euler (deg)
        ox, oy, oz, ow = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        r, p, y = quat_to_euler_deg(ox, oy, oz, ow)
        
        r_m, p_m, y_m = self._apply_axis_map(r, p, y)

        roll_carla  = self._base_rot.roll  + (r_m * self._rotation_mult)
        pitch_carla = self._base_rot.pitch + (p_m * self._rotation_mult)
        yaw_carla   = self._base_rot.yaw   + (y_m * self._rotation_mult)

        new_rot = carla.Rotation(pitch=pitch_carla, yaw=yaw_carla, roll=roll_carla)
        self._camera.set_transform(carla.Transform(self._base_loc, new_rot))
        
    
        
    
def ros_background_spin(camera, base_transform):
    rclpy.init()
    node = CarlaRosNode(camera, base_transform)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    t = threading.Thread(target=executor.spin, daemon=True)
    t.start()
    return executor, node, t
        
        

def main(img_queue, img_lock):
    parser = argparse.ArgumentParser(description="Publish vehicle data and control the vehicle with ROS")
    parser.add_argument('--port', type=int, default=2000, help="CARLA port")
    parser.add_argument('--host', type=str, default='localhost', help="CARLA port")
    args = parser.parse_args()
    
    # CARLA Client setup
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_library = world.get_blueprint_library()
    
    vehicle_bp = bp_library.find('vehicle.tesla.model3')
    vehicle = world.spawn_actor(vehicle_bp,
                                np.random.choice(world.get_map().get_spawn_points()))
    # ================
    # Comment for manual control

    #tm = client.get_trafficmanager()
    vehicle.set_autopilot(True)

    # =================
    print(f"Spawned {vehicle.type_id}")

    # Attach a camera RGB sensor to the side of the car
    cam_bp = bp_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '480')
    cam_transform = carla.Transform(carla.Location(x=0.8, y=1.2, z=1.0), carla.Rotation(yaw=15))
    cam = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    print(f"Spawned {cam.type_id}")
    
    # When camera gets an image, pass it to CV2 process
    def cam_callback(img):
        with img_lock:
            img_queue.put((bytes(img.raw_data), img.width, img.height))
    
    cam.listen(cam_callback)
    
    executor, node, ros_t = ros_background_spin(cam, cam_transform)
        
    print(f"Waiting for sensors to start...")
    time.sleep(5)
    print(f"Reading ") 
    
    try:
        # Main blocking loop
        while rclpy.ok():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            print("Trying to exit in a controlled way")
            cam.stop()
            cam.destroy()
            vehicle.destroy()
            executor.shutdown(timeout_sec=1.0)
            node.destroy_node()
            rclpy.shutdown()
            with img_lock:
                img_queue.put(None)
        except Exception:
            print(f"Error exiting correctly!")
            

if __name__ == '__main__':
    
    img_queue = multiprocessing.Queue()
    img_lock = multiprocessing.Lock()
    
    processes = [
        multiprocessing.Process(target=main, args=(img_queue, img_lock)),
        multiprocessing.Process(target=dash_cam, args=(img_queue, img_lock))
    ]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()

