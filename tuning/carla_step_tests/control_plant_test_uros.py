"""
ROS2 node that simulates a discrete plant for HIL-style PID testing.

Outputs
  - ROS topics and optional plots.

Run (example)
  python3 tuning/carla_step_tests/control_plant_test_uros.py
"""

import numpy as np
import matplotlib.pyplot as plt
import control as ctl
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from pid_message.msg import PID
import argparse
import time
from threading import Thread
import sys
from lib.CarlaROSNode import CarlaROSNode

class CarlaROSNode(Node):
    def __init__(self, args):
        super().__init__('carla_ros_node')
        self.get_logger().info("Carla ROS Node started (real-time mode).")
        self._args = args
        # QoS profiles
        qos_params = rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE)
        qos_stream = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT)
        # Publishers
        self._pv_pub  = self.create_publisher(Float64, 'pv',  qos_stream)
        self._pid_param_pub = self.create_publisher(PID, 'pid_params', qos_params)
        # Subscribers
        self._u_sub = self.create_subscription(Float64, 'u', self.u_cb, qos_stream)
        self._r_sub = self.create_subscription(Float64, 'r', self.r_cb, qos_params)
        # Internal signals
        self._r_val = 0.0
        self._u_val = 0.0
        self._y_val = 0.0
        # Timing
        self.Ts = args.Ts
        self.k_step = 0
        self.last_step_monotonic = time.perf_counter()

        # ------------------------------------------------------------------
        # Discrete plant (same as pure-python simulation: state-space form)
        # ------------------------------------------------------------------
        Ts = self.Ts
        s = ctl.TransferFunction.s
        G0 = (8.7129*s + 0.05263) / (s**2 + 0.1953*s + 0.0001874) 
        Gd = ctl.c2d(G0, Ts, method='tustin')

        # Convert to discrete state-space (same as in control_plant_test_pure_python.py)
        Ad, Bd, Cd, Dd = ctl.ssdata(ctl.ss(Gd))
        self.Ad = np.asarray(Ad)
        self.Bd = np.asarray(Bd).reshape(-1)
        self.Cd = np.asarray(Cd)
        self.Dd = np.asarray(Dd).reshape(-1)

        # Initial plant state
        self.x = np.zeros(self.Ad.shape[0])

        # ------------------------------------------------------------------
        # History (for plotting)
        # ------------------------------------------------------------------
        self.enable_plot = not args.no_plot
        self.max_points = args.max_points
        self.t_hist = []
        self.r_hist = []
        self.y_hist = []
        self.u_hist = []
        if self.enable_plot:
            plt.ion()
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
            self.line_r, = self.ax1.plot([], [], '--', label='r')
            self.line_y, = self.ax1.plot([], [], label='y')
            self.ax1.grid(True); self.ax1.legend()
            self.line_u, = self.ax2.plot([], [], label='u')
            self.ax2.grid(True); self.ax2.legend()
            self.ax2.set_xlabel('t [s]')
            self.fig.tight_layout()
        # Publish PID params once
        self.publish_params()
        # High-rate plant timer (lightweight)
        self.plant_timer = self.create_timer(0.01, self.step_callback)
        # Plot handled externally in main loop thread

    def publish_params(self):
        m = PID()
        m.kp = self._args.kp
        m.ki = self._args.ki
        m.kd = self._args.kd
        m.n  = self._args.N
        self._pid_param_pub.publish(m)
        self.get_logger().info(f"PID params published: {m}")

    def u_cb(self, msg: Float64):
        self._u_val = float(msg.data)

    def r_cb(self, msg: Float64):
        self._r_val = float(msg.data)

    def step_callback(self):
        # Current control from microcontroller PID
        u_curr = self._u_val

        # State-space update 
        self.x = self.Ad @ self.x + self.Bd * u_curr
        y_curr = (self.Cd @ self.x + self.Dd * u_curr).item()

        self._y_val = y_curr
        pv_msg = Float64()
        pv_msg.data = float(y_curr)
        self._pv_pub.publish(pv_msg)

        t = self.k_step * self.Ts
        self.k_step += 1

        # Store history
        self.t_hist.append(t)
        self.r_hist.append(self._r_val)
        self.y_hist.append(y_curr)
        self.u_hist.append(u_curr)
        if self.max_points and len(self.t_hist) > self.max_points:
            trim = len(self.t_hist) - self.max_points
            self.t_hist = self.t_hist[trim:]
            self.r_hist = self.r_hist[trim:]
            self.y_hist = self.y_hist[trim:]
            self.u_hist = self.u_hist[trim:]

        # Jitter logging (optional)
        if self._args.jitter_log and (self.k_step % self._args.jitter_log_interval == 0):
            now = time.perf_counter()
            dt = now - self.last_step_monotonic
            self.last_step_monotonic = now
            self.get_logger().info(f"step dt={dt*1000:.2f} ms (target {self.Ts*1000:.2f} ms)")

    def update_plot(self):
        if not self.enable_plot or not self.t_hist:
            return
        t = self.t_hist[-1]
        self.line_r.set_data(self.t_hist, self.r_hist)
        self.line_y.set_data(self.t_hist, self.y_hist)
        self.line_u.set_data(self.t_hist, self.u_hist)
        # Non-scrolling: fixed 0 .. x_window (hold) or expand 0 .. max(x_window, t)
        if self._args.hold_x:
            x_max = self._args.x_window
        else:
            x_max = max(self._args.x_window, t)
        self.ax1.set_xlim(0, x_max)
        self.ax2.set_xlim(0, x_max)
        if self._args.auto_scale and len(self.y_hist) > 5:
            y_min = min(min(self.r_hist), min(self.y_hist))
            y_max = max(max(self.r_hist), max(self.y_hist))
            self.ax1.set_ylim(y_min - 0.05 * abs(y_min + 1e-9),
                              y_max + 0.05 * abs(y_max + 1e-9))
            if len(self.u_hist) > 5:
                u_min = min(self.u_hist); u_max = max(self.u_hist)
                self.ax2.set_ylim(u_min - 0.05 * abs(u_min + 1e-9),
                                  u_max + 0.05 * abs(u_max + 1e-9))
        self.fig.canvas.draw_idle()
        if hasattr(self.fig.canvas, "flush_events"):
            self.fig.canvas.flush_events()

def main():
    parser = argparse.ArgumentParser(description="Real-time plant simulation interfaced with external PID (STM32).")
    parser.add_argument('--kp', type=float, default=0.47446807)
    parser.add_argument('--ki', type=float, default=0.15472707)
    parser.add_argument('--kd', type=float, default=7.83353195)
    parser.add_argument('--N', type=float, default=20.0)
    parser.add_argument('--Ts', type=float, default=0.01, help="Sampling period (s)")
    parser.add_argument('--plot-period', type=float, default=0.1, help="Plot update period (s)")
    parser.add_argument('--no-plot', action='store_true', help="Disable live plotting")
    parser.add_argument('--plot-decimation', type=int, default=5, help="(Unused now) kept for compatibility")
    parser.add_argument('--x-window', type=float, default=20.0, help="Initial horizontal window (s)")
    parser.add_argument('--hold-x', action='store_true', help="Keep x-axis fixed at x-window (no expansion)")
    parser.add_argument('--auto-scale', action='store_true', help="Enable dynamic Y scaling")
    parser.add_argument('--max-points', type=int, default=0, help="Limit stored points (0 = unlimited)")
    parser.add_argument('--jitter-log', action='store_true', help="Log timer period jitter")
    parser.add_argument('--jitter-log-interval', type=int, default=50, help="Samples between jitter logs")
    args = parser.parse_args()

    rclpy.init()
    node = CarlaROSNode(args)

    # Run ROS executor in background thread
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    exec_thread = Thread(target=executor.spin, daemon=True)
    exec_thread.start()

    last_plot = time.perf_counter()
    try:
        while rclpy.ok():
            if node.enable_plot:
                now = time.perf_counter()
                if now - last_plot >= args.plot_period:
                    node.update_plot()
                    last_plot = now
                    plt.pause(0.001)
            time.sleep(0.002)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupt received, shutting down.")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        if node.enable_plot:
            plt.ioff()
            node.update_plot()
            plt.show()
        sys.exit(0)

if __name__ == '__main__':
    main()
