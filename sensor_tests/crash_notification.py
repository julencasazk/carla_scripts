#!/usr/bin/env python3
"""
Prototype ROS2 node for IMU-based crash notification (stub).

Outputs
  - Prints received IMU messages.

Run (example)
  python3 sensor_tests/crash_notification.py
"""

import numpy as np
import rclpy
import carla
import argparse
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32
from std_msgs.msg import Bool
from sensor_msgs.msg import Imu
import queue
import threading

class CarlaROSCrashNode(Node):
    def __init__(self):
        #TODO
        qos = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT)
        self._imu_subscription = self.create_subscription(Imu, 'imu_data', self.imu_callback, qos)
    
    def imu_callback(self, msg : Imu):
        print(msg)


def p_carla(args):
    #TODO
    print("Nada")
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="CARLA ROS2 Crash Notification Node")
    parser.add_argument('--host', type=str, help="CARLA host", default='localhost')
    parser.add_argument('--port', type=int, help="CARLA port", default=2000)
    
    rclpy.init()
    node = CarlaROSCrashNode()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    t = threading.Thread(target=executor.spin, daemon=True)
    t.start()
    t.join() 
