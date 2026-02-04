"""
Measures time to first throttle command after publishing state topics.

Outputs
  - Console logs of startup latency.

Run (example)
  python3 tools/ego_link_startup_latency_debug.py --rate-hz 100
"""

import argparse
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32


class LinkStartupLatency(Node):
    def __init__(
        self,
        rate_hz: float,
        speed: float,
        dist: float,
        setpoint: float,
        throttle_topic: str,
        speed_topic: str,
        dist_topic: str,
        setpoint_topic: str,
        timeout_s: float,
        log_every: int,
    ) -> None:
        super().__init__("ego_link_startup_latency_debug")

        self.period_s = 1.0 / float(rate_hz)
        self.speed = float(speed)
        self.dist = float(dist)
        self.setpoint = float(setpoint)
        self.timeout_s = float(timeout_s)
        self.log_every = int(log_every)

        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.pub_speed = self.create_publisher(Float32, speed_topic, qos)
        self.pub_dist = self.create_publisher(Float32, dist_topic, qos)
        self.pub_setpoint = self.create_publisher(Float32, setpoint_topic, qos)

        self.create_subscription(Float32, throttle_topic, self._throttle_cb, qos)

        self._t0_wall_s = time.perf_counter()
        self._first_throttle_wall_s: Optional[float] = None
        self._pub_count = 0
        self._last_pub_wall_s: Optional[float] = None

        self.create_timer(self.period_s, self._tick)

        if self.timeout_s > 0.0:
            self.create_timer(0.05, self._timeout_check)

        self.get_logger().info(
            "Publishing speed/dist/setpoint and timing until first throttle is received."
        )
        self.get_logger().info(
            f"Topics: speed={speed_topic}, dist={dist_topic}, setpoint={setpoint_topic}, throttle_in={throttle_topic}"
        )
        self.get_logger().info(
            f"rate={rate_hz:.1f}Hz, speed={self.speed}, dist={self.dist}, setpoint={self.setpoint}, timeout={self.timeout_s}s"
        )

    def _throttle_cb(self, msg: Float32) -> None:
        if self._first_throttle_wall_s is not None:
            return
        self._first_throttle_wall_s = time.perf_counter()
        dt_s = self._first_throttle_wall_s - self._t0_wall_s
        self.get_logger().info(f"First throttle received: {float(msg.data):.6f}")
        self.get_logger().info(f"Startup latency (wall): {dt_s:.6f} s")
        raise SystemExit(0)

    def _tick(self) -> None:
        now_s = time.perf_counter()
        if self._last_pub_wall_s is not None and self.log_every > 0 and (self._pub_count % self.log_every) == 0:
            dt_s = now_s - self._last_pub_wall_s
            hz = (1.0 / dt_s) if dt_s > 0.0 else float("inf")
            self.get_logger().info(f"pub_timer_dt={dt_s:.6f}s (~{hz:.1f} Hz), target={self.period_s:.6f}s")
        self._last_pub_wall_s = now_s
        self._pub_count += 1

        self.pub_speed.publish(Float32(data=float(self.speed)))
        self.pub_dist.publish(Float32(data=float(self.dist)))
        self.pub_setpoint.publish(Float32(data=float(self.setpoint)))

    def _timeout_check(self) -> None:
        if self._first_throttle_wall_s is not None:
            return
        elapsed_s = time.perf_counter() - self._t0_wall_s
        if elapsed_s >= self.timeout_s:
            self.get_logger().error(f"Timeout waiting for first throttle after {elapsed_s:.3f}s")
            raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish constant ego state topics and measure time until first throttle is received (no CARLA)."
    )
    parser.add_argument("--rate-hz", type=float, default=100.0, help="Publish rate (Hz)")
    parser.add_argument("--speed", type=float, default=0.0, help="veh_ego/state/speed value (m/s)")
    parser.add_argument("--dist", type=float, default=100.0, help="veh_ego/state/dist_to_veh value (m)")
    parser.add_argument("--setpoint", type=float, default=33.33, help="veh_ego/state/setpoint value (m/s)")
    parser.add_argument("--timeout-s", type=float, default=30.0, help="Timeout waiting for throttle (0 disables)")
    parser.add_argument("--log-every", type=int, default=100, help="Log publish timer dt every N callbacks")
    parser.add_argument("--throttle-topic", type=str, default="veh_ego/command/throttle", help="Throttle input topic")
    parser.add_argument("--speed-topic", type=str, default="veh_ego/state/speed", help="Speed output topic")
    parser.add_argument("--dist-topic", type=str, default="veh_ego/state/dist_to_veh", help="Distance output topic")
    parser.add_argument("--setpoint-topic", type=str, default="veh_ego/state/setpoint", help="Setpoint output topic")
    args = parser.parse_args()

    rclpy.init()
    node = LinkStartupLatency(
        rate_hz=float(args.rate_hz),
        speed=float(args.speed),
        dist=float(args.dist),
        setpoint=float(args.setpoint),
        throttle_topic=str(args.throttle_topic),
        speed_topic=str(args.speed_topic),
        dist_topic=str(args.dist_topic),
        setpoint_topic=str(args.setpoint_topic),
        timeout_s=float(args.timeout_s),
        log_every=int(args.log_every),
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
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
