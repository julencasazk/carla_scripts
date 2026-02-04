#!/usr/bin/env python3
"""
Logger for MCU diagnostic topics to CSV.

Outputs
  - CSV file with per-field diagnostics and receive ages.

Run (example)
  python3 tools/mcu_diag_logger.py --vehicle veh_0 --out diag.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from typing import Dict, Optional

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import UInt32


DIAG_FIELDS = [
    "free_heap_bytes",
    "min_ever_free_heap_bytes",
    "uros_stack_free_bytes",
    "ctrl_stack_free_bytes",
    "microros_alloc_count",
    "microros_free_count",
    "microros_realloc_count",
    "microros_calloc_count",
    "microros_used_bytes",
    "microros_total_allocated_bytes",
]


def _topic(vehicle: str, field: str) -> str:
    v = str(vehicle).strip().strip("/")
    return f"/{v}/diag/{field}"


class McuDiagLogger(Node):
    def __init__(self, vehicle: str, out_csv: str, rate_hz: float, append: bool) -> None:
        super().__init__("mcu_diag_logger")

        self._vehicle = str(vehicle)
        self._out_csv = str(out_csv)
        self._rate_hz = float(rate_hz)
        self._append = bool(append)

        self._start_wall_s = time.perf_counter()
        self._latest: Dict[str, Optional[int]] = {k: None for k in DIAG_FIELDS}
        self._latest_rx_wall_s: Dict[str, Optional[float]] = {k: None for k in DIAG_FIELDS}

        # Match the rest of the project: BEST_EFFORT + VOLATILE.
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        for field in DIAG_FIELDS:
            topic = _topic(self._vehicle, field)
            self.create_subscription(
                UInt32,
                topic,
                lambda msg, f=field: self._cb(msg, f),
                qos,
            )

        mode = "a" if self._append else "w"
        self._csv_file = open(self._out_csv, mode=mode, newline="")
        self._csv_writer = csv.writer(self._csv_file)

        if not self._append:
            header = ["wall_time_s", "ros_time_s"]
            header.extend([f"{k}" for k in DIAG_FIELDS])
            header.extend([f"{k}_rx_age_ms" for k in DIAG_FIELDS])
            self._csv_writer.writerow(header)
            self._csv_file.flush()

        period_s = (1.0 / self._rate_hz) if self._rate_hz > 0.0 else 1.0
        self._timer = self.create_timer(float(period_s), self._tick)

        self.get_logger().info(
            f"Logging MCU diag for {self._vehicle} to {self._out_csv} at {self._rate_hz:.2f} Hz"
        )

    def _cb(self, msg: UInt32, field: str) -> None:
        self._latest[field] = int(msg.data)
        self._latest_rx_wall_s[field] = time.perf_counter()

    def _tick(self) -> None:
        now_wall = time.perf_counter()
        wall_time_s = float(now_wall - self._start_wall_s)
        ros_now = self.get_clock().now()
        ros_time_s = float(ros_now.nanoseconds) * 1e-9

        row = [f"{wall_time_s:.6f}", f"{ros_time_s:.6f}"]
        for k in DIAG_FIELDS:
            v = self._latest.get(k)
            row.append("" if v is None else str(int(v)))
        for k in DIAG_FIELDS:
            rx_s = self._latest_rx_wall_s.get(k)
            if rx_s is None:
                row.append("")
            else:
                row.append(f"{(now_wall - float(rx_s)) * 1000.0:.3f}")

        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        try:
            self._csv_file.close()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Log micro-ROS MCU diagnostic UInt32 topics to CSV.")
    ap.add_argument("--vehicle", type=str, default="veh_3", help="Vehicle namespace (default: veh_3).")
    ap.add_argument("-o", "--out", type=str, default="mcu_diag.csv", help="Output CSV path.")
    ap.add_argument("--rate-hz", type=float, default=1.0, help="Logging rate (default: 1.0 Hz).")
    ap.add_argument("--append", action="store_true", help="Append to CSV instead of overwriting.")
    args = ap.parse_args()

    rclpy.init()
    node = McuDiagLogger(
        vehicle=str(args.vehicle),
        out_csv=str(args.out),
        rate_hz=float(args.rate_hz),
        append=bool(args.append),
    )

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.remove_node(node)
        except Exception:
            pass
        node.close()
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
