"""
ROS2 node that prints received throttle commands for quick debugging.

Outputs
  - Console logs of throttle values.

Run (example)
  python3 tools/throttle_debug_node.py
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

class ThrottleDebug(Node):
    def __init__(self):
        super().__init__('throttle_debug')
        self.sub = self.create_subscription(
            Float32,
            '/veh_lead/command/throttle',
            self.cb,
            10,
        )
        self.get_logger().info('ThrottleDebug node started, listening on /veh_lead/command/throttle')

    def cb(self, msg: Float32):
        self.get_logger().info(f'[ThrottleDebug] received: {msg.data:.2f}')

def main():
    rclpy.init()
    node = ThrottleDebug()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
