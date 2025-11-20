import numpy as np
import matplotlib.pyplot as plt
import control as ctl
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import argparse
import time

class CarlaROSNode(Node):
    def __init__(self, args):
        super().__init__('carla_ros_node')
        self.get_logger().info("Carla ROS Node started (real-time mode).")
        self._args = args

        # QoS profiles
        qos_params = rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE)
        qos_stream = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT)

        # Publishers
        self._pv_pub  = self.create_publisher(Float32, 'pv',  qos_stream)
        self._kp_pub  = self.create_publisher(Float32, 'kp',  qos_params)
        self._ki_pub  = self.create_publisher(Float32, 'ki',  qos_params)
        self._kd_pub  = self.create_publisher(Float32, 'kd',  qos_params)
        self._N_pub   = self.create_publisher(Float32, 'N',   qos_params)
        self._kaw_pub = self.create_publisher(Float32, 'kaw', qos_params)

        # Subscribers
        self._u_sub = self.create_subscription(Float32, 'u', self.u_cb, qos_stream)
        self._r_sub = self.create_subscription(Float32, 'r', self.r_cb, qos_params)

        # Internal signals
        self._r_val = 0.0
        self._u_val = 0.0
        self._y_val = 0.0

        # Timing
        self.Ts = args.Ts
        self.start_monotonic = time.monotonic()
        self.k_step = 0

        # Build discrete plant
        k = 0.156
        wn = 0.396
        zeta = 0.661
        Td = 0.146

        s = ctl.TransferFunction.s
        G0 = k / (s**2 + 2*zeta*wn*s + wn**2)
        num_delay, den_delay = ctl.pade(Td, 1)
        Gc = G0 * ctl.tf(num_delay, den_delay)
        Gd = ctl.c2d(Gc, self.Ts, method='tustin')
        numd, dend = ctl.tfdata(Gd)
        numd = np.squeeze(numd)
        dend = np.squeeze(dend)
        self.a = dend / dend[0]
        self.b = numd / dend[0]
        self.na = len(self.a) - 1
        self.nb = len(self.b)
        self.y_prev = np.zeros(self.na)
        self.u_prev = np.zeros(self.nb)

        # Plot / history
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
            self.ax1.grid(True)
            self.ax1.legend()
            self.line_u, = self.ax2.plot([], [], label='u')
            self.ax2.grid(True)
            self.ax2.legend()
            self.ax2.set_xlabel('t [s]')
            self.fig.tight_layout()

        # Publish PID params once
        self.publish_params()

        # Create simulation timer
        self.timer = self.create_timer(self.Ts, self.step_callback)

    def publish_params(self):
        msgs = {
            'kp': self._args.kp,
            'ki': self._args.ki,
            'kd': self._args.kd,
            'N':  self._args.N,
            'kaw': self._args.kaw
        }
        for topic, value in msgs.items():
            m = Float32()
            m.data = float(value)
            getattr(self, f'_{topic}_pub').publish(m)
        self.get_logger().info(f"PID params published: {msgs}")

    def u_cb(self, msg: Float32):
        self._u_val = float(msg.data)

    def r_cb(self, msg: Float32):
        self._r_val = float(msg.data)

    def step_callback(self):
        # Current controller output
        u_curr = self._u_val

        # Update u history
        self.u_prev = np.roll(self.u_prev, 1)
        self.u_prev[0] = u_curr

        # Difference equation
        y_curr = -np.dot(self.a[1:], self.y_prev) + np.dot(self.b, self.u_prev)

        # Update y history
        self.y_prev = np.roll(self.y_prev, 1)
        self.y_prev[0] = y_curr
        self._y_val = y_curr

        # Publish pv
        pv_msg = Float32()
        pv_msg.data = float(y_curr)
        self._pv_pub.publish(pv_msg)

        # Time
        t = self.k_step * self.Ts
        self.k_step += 1

        # Append histories
        self.t_hist.append(t)
        self.r_hist.append(self._r_val)
        self.y_hist.append(y_curr)
        self.u_hist.append(u_curr)

        # Trim history if needed
        if self.max_points and len(self.t_hist) > self.max_points:
            trim = len(self.t_hist) - self.max_points
            self.t_hist = self.t_hist[trim:]
            self.r_hist = self.r_hist[trim:]
            self.y_hist = self.y_hist[trim:]
            self.u_hist = self.u_hist[trim:]

        # Update plot (decimate for speed)
        if self.enable_plot and (self.k_step % self._args.plot_decimation == 0):
            self.line_r.set_data(self.t_hist, self.r_hist)
            self.line_y.set_data(self.t_hist, self.y_hist)
            self.line_u.set_data(self.t_hist, self.u_hist)
            # Rescale axes
            self.ax1.set_xlim(max(0, t - self._args.x_window), t + 0.01)
            self.ax2.set_xlim(max(0, t - self._args.x_window), t + 0.01)
            if self._args.auto_scale:
                if len(self.y_hist) > 5:
                    y_min = min(min(self.r_hist), min(self.y_hist))
                    y_max = max(max(self.r_hist), max(self.y_hist))
                    self.ax1.set_ylim(y_min - 0.05 * abs(y_min + 1e-9),
                                      y_max + 0.05 * abs(y_max + 1e-9))
                if len(self.u_hist) > 5:
                    u_min = min(self.u_hist)
                    u_max = max(self.u_hist)
                    self.ax2.set_ylim(u_min - 0.05 * abs(u_min + 1e-9),
                                      u_max + 0.05 * abs(u_max + 1e-9))
            self.fig.canvas.draw()
            plt.pause(0.001)

def main():
    parser = argparse.ArgumentParser(description="Real-time plant simulation interfaced with external PID (STM32).")
    parser.add_argument('--kp', type=float, default=1.5) # 1.5
    parser.add_argument('--ki', type=float, default=0.3) # 0.3
    parser.add_argument('--kd', type=float, default=0.0) # 20.0
    parser.add_argument('--N', type=float, default=10.0) # 15.0
    parser.add_argument('--kaw', type=float, default=1.0) # 1.0
    parser.add_argument('--Ts', type=float, default=0.01, help="Sampling period (s)")
    parser.add_argument('--no-plot', action='store_true', help="Disable live plotting")
    parser.add_argument('--plot-decimation', type=int, default=5, help="Update plot every N samples")
    parser.add_argument('--x-window', type=float, default=10.0, help="Time window (s) visible on plot")
    parser.add_argument('--auto-scale', action='store_true', help="Enable dynamic Y scaling")
    parser.add_argument('--max-points', type=int, default=0, help="Limit stored points (0 = unlimited)")
    args = parser.parse_args()

    rclpy.init()
    node = CarlaROSNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        rclpy.shutdown()
        if node.enable_plot:
            plt.ioff()
            plt.show()

if __name__ == '__main__':
    main()
