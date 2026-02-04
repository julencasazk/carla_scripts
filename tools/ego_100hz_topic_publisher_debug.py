"""
Publishes synthetic ego speed/dist/setpoint at high rate (no CARLA required).

Outputs
  - ROS topics at configured rate.

Run (example)
  python3 tools/ego_100hz_topic_publisher_debug.py --rate-hz 100
"""

import argparse
import random
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32


class EgoTopicPublisher(Node):
    def __init__(
        self,
        rate_hz: float,
        mode: str,
        speed: float,
        dist: float,
        setpoint: float,
        rand_min: float,
        rand_max: float,
        log_every: int,
    ) -> None:
        super().__init__("ego_100hz_topic_publisher_debug")
        self.period_s = 1.0 / float(rate_hz)
        self.mode = str(mode)
        self.const_speed = float(speed)
        self.const_dist = float(dist)
        self.const_setpoint = float(setpoint)
        self.rand_min = float(rand_min)
        self.rand_max = float(rand_max)
        self.log_every = int(log_every)

        qos_state = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.pub_speed = self.create_publisher(Float32, "veh_ego/state/speed", qos_state)
        self.pub_dist = self.create_publisher(Float32, "veh_ego/state/dist_to_veh", qos_state)
        self.pub_setpoint = self.create_publisher(Float32, "veh_ego/state/setpoint", qos_state)

        self._last_wall_s: Optional[float] = None
        self._count = 0
        self.create_timer(self.period_s, self._tick)

    def _sample(self, constant_value: float) -> float:
        if self.mode == "random":
            return random.uniform(self.rand_min, self.rand_max)
        return constant_value

    def _tick(self) -> None:
        now_s = time.perf_counter()
        if self._last_wall_s is not None and self.log_every > 0 and (self._count % self.log_every) == 0:
            dt_s = now_s - self._last_wall_s
            hz = (1.0 / dt_s) if dt_s > 0.0 else float("inf")
            self.get_logger().info(f"timer_dt={dt_s:.6f}s (~{hz:.1f} Hz), target={self.period_s:.6f}s")
        self._last_wall_s = now_s
        self._count += 1

        self.pub_speed.publish(Float32(data=float(self._sample(self.const_speed))))
        self.pub_dist.publish(Float32(data=float(self._sample(self.const_dist))))
        self.pub_setpoint.publish(Float32(data=float(self._sample(self.const_setpoint))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug: publish veh_ego speed/dist/setpoint at configurable Hz (no CARLA)."
    )
    parser.add_argument("--rate-hz", type=float, default=100.0, help="Publish rate (Hz)")
    parser.add_argument(
        "--mode",
        choices=["constant", "random"],
        default="constant",
        help="Publish constant values or random values per tick.",
    )
    parser.add_argument("--speed", type=float, default=0.0, help="Constant speed value (m/s)")
    parser.add_argument("--dist", type=float, default=0.0, help="Constant dist value (m)")
    parser.add_argument("--setpoint", type=float, default=0.0, help="Constant setpoint value (m/s)")
    parser.add_argument("--rand-min", type=float, default=0.0, help="Random range min (for all topics)")
    parser.add_argument("--rand-max", type=float, default=1.0, help="Random range max (for all topics)")
    parser.add_argument("--log-every", type=int, default=100, help="Log timer dt every N callbacks")
    args = parser.parse_args()

    rclpy.init()
    node = EgoTopicPublisher(
        rate_hz=float(args.rate_hz),
        mode=str(args.mode),
        speed=float(args.speed),
        dist=float(args.dist),
        setpoint=float(args.setpoint),
        rand_min=float(args.rand_min),
        rand_max=float(args.rand_max),
        log_every=int(args.log_every),
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
