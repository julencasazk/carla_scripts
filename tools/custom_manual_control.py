#!/usr/bin/env python
"""
Keyboard-driven manual control node that publishes throttle/brake/steer.

Outputs
  - ROS topics with manual command values.

Run (example)
  python3 tools/custom_manual_control.py


# From Simon D. Levy
"""

from lib import kb_controller
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from std_msgs.msg import Bool
import threading
import numpy as np

ESP = False

class VehicleControlNode(Node):
    
    def __init__(self, keypress_buff):
        super().__init__('vehicle_ctrl')
        self._keypress_buff = keypress_buff
        qos = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT)
        if ESP:
            self._throttle_pub = self.create_publisher(Float32, 'throttle_debug', qos)
            self._brake_pub = self.create_publisher(Float32, 'brake_debug', qos)
            self._steer_pub = self.create_publisher(Float32, 'steer_debug', qos)
            self._reverse_pub = self.create_publisher(Bool, 'reverse_debug', qos)
        else:
            self._throttle_pub = self.create_publisher(Float32, 'throttle_cmd', qos)
            self._brake_pub = self.create_publisher(Float32, 'brake_cmd', qos)
            self._steer_pub = self.create_publisher(Float32, 'steer_cmd', qos)
            self._reverse_pub = self.create_publisher(Bool, 'reverse_cmd', qos)
        self._timer = self.create_timer(0.01, self.timer_cb)
        self._return_timer = self.create_timer(0.1, self.return_timer_cb)
        self._throttle = Float32()
        self._brake = Float32()
        self._steer = Float32()
        self._reverse = Bool()
        self._return_strength = 0.1
        self._cmd_strength = 0.3

    def return_timer_cb(self):
        self._throttle.data -= self._return_strength
        if (self._throttle.data < 0.0):
            self._throttle.data = 0.0
        
        self._brake.data -= self._return_strength
        if (self._brake.data < 0.0):
            self._brake.data = 0.0 
            
        if (self._steer.data > 0):
            if (self._steer.data > self._return_strength):
                self._steer.data -= self._return_strength
            else:
                self._steer.data = 0.0

        elif (self._steer.data < 0):
            if (np.abs(self._steer.data) > self._return_strength):
                self._steer.data += self._return_strength
            else:
                self._steer.data = 0.0

    def timer_cb(self):
        try:
            c = self._keypress_buff.pop()

            if ord(c) == ord("w"):
                if (self._reverse.data == True and self._throttle.data == 0):
                    self._reverse.data = False
                else:
                    if (not self._reverse.data):
                        self._throttle.data += self._cmd_strength
                    else:
                        self._brake.data += self._cmd_strength

                    if (self._throttle.data > 1.0):
                        self._throttle.data = 1.0
                    if (self._brake.data > 1.0):
                        self._brake.data = 1.0
                    
                
            elif ord(c) == ord("s"):
                if (self._reverse.data == False and self._throttle.data == 0):
                    self._reverse.data = True
                else:
                    if (self._reverse.data):
                        self._throttle.data += self._cmd_strength
                    else:
                        self._brake.data += self._cmd_strength
                        
                    if (self._throttle.data > 1.0):
                        self._throttle.data = 1.0
                    if (self._brake.data > 1.0):
                        self._brake.data = 1.0

            elif ord(c) == ord("a"):
                self._steer.data -= self._cmd_strength
                if (self._steer.data < -1.0):
                    self._steer.data = -1.0
                    
            elif ord(c) == ord("d"):
                self._steer.data += self._cmd_strength
                if (self._steer.data > 1.0):
                    self._steer.data = 1.0         

                
        except IndexError:
            # Emtpy array
            pass

        self._throttle_pub.publish(self._throttle)
        self._steer_pub.publish(self._steer)
        self._reverse_pub.publish(self._reverse)


def main():
    rclpy.init()
    kb = kb_controller.KBHit()
    print("Hit any key, or ESC to exit")
    chars = []    
    node = VehicleControlNode(chars)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    t = threading.Thread(target=executor.spin, daemon=True)
    t.start()
    while True:
        if kb.kbhit():
            c = kb.getch()
            # Keep a circular buffer of max 5 chars (drop oldest)
            if len(chars) >= 5:
                del chars[0]
            chars.append(c)
            if ord(c) == 27:
                executor.shutdown(timeout_sec=1.0)
                node.destroy_node()
                rclpy.shutdown()
                break
    kb.set_normal_term()
    
if __name__ == "__main__":
    main()
