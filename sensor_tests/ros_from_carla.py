#! /usr/bin/env python3
"""
Simple CARLA-to-ROS bridge for manual control testing.

Outputs
  - ROS topics: speed publisher and throttle/steer/brake subscribers.

Run (example)
  python3 sensor_tests/ros_from_carla.py --host localhost --port 2000
"""

import rclpy
import carla
import numpy as np
import cv2
import argparse
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32
from std_msgs.msg import Bool
import queue
import multiprocessing
import time
import threading


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

class CarlaRosNode(Node):
    def __init__(self, vehicle):
        super().__init__('carla_vehicle_control')
        self._vehicle = vehicle
        qos = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT)
        self._speed_pub = self.create_publisher(Float32, 'speed', qos)
        self._throttle_sub = self.create_subscription(Float32, 'throttle_cmd', self.throttle_cb, qos)
        self._steer_sub = self.create_subscription(Float32, 'steer_cmd', self.steer_cb, qos)
        self._steer_sub = self.create_subscription(Bool, 'reverse_cmd', self.reverse_cb, qos)
        self._brake_sub = self.create_subscription(Float32, 'brake_cmd', self.throttle_cb, qos)
        self._pub_timer = self.create_timer(0.1, self.publish)
        
    def publish(self):
        speed_msg = Float32()
        vel = self._vehicle.get_velocity()
        speed = float(np.linalg.norm([vel.x, vel.y, vel.z]))
        speed_msg.data = 3.6 * speed
        self._speed_pub.publish(speed_msg)
    
    def throttle_cb(self, msg: Float32):
        control = self._vehicle.get_control()
        control.throttle = msg.data
        self._vehicle.apply_control(control)
    
    def steer_cb(self, msg: Float32):
        control = self._vehicle.get_control()
        control.steer = msg.data
        self._vehicle.apply_control(control)
        
    def reverse_cb(self, msg: Bool):
        control = self._vehicle.get_control()
        control.reverse = msg.data
        self._vehicle.apply_control(control)
        
    def brake_cb(self, msg: Float32):
        control = self._vehicle.get_control()
        control.brake = msg.data
        self._vehicle.apply_control(control)
        
    
def ros_background_spin(vehicle):
    rclpy.init()
    node = CarlaRosNode(vehicle)     
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
    #vehicle.set_autopilot(True, tm.get_port())

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
    
    executor, node, ros_t = ros_background_spin(vehicle)
        
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
