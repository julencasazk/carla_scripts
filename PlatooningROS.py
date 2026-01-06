"""
ROS2 platoon member node, CARLA-agnostic.

This node:
  * Subscribes to:
      <name>/state/speed       : Float32, current ego speed [m/s]
      <name>/state/setpoint    : Float32, base (global) speed setpoint [m/s]
      <name>/state/dist_to_veh : Float32, distance to preceding vehicle [m]
  * Publishes:
      <name>/command/throttle  : Float32, [0..1]
      <name>/command/brake     : Float32, [0..1]

All vehicle and simulator specifics (e.g. CARLA VehicleControl, transforms)
must be handled in a separate bridge node.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32, Bool

from PID import PID


class PlatoonMember(Node):
    def __init__(
        self,
        name: str,
        role: str,
        slow_pid: PID,
        mid_pid: PID,
        fast_pid: PID,
        desired_time_headway: float = 0.5,
        min_spacing: float = 5.0,
        K_dist: float = 0.2,
        control_period: float = 0.05,
        platoon_id: str = "plat_0",
    ):
        """
        role: "lead" or "follower"
        For follower, a distance-based correction is applied to the speed setpoint.
        """
        super().__init__(name)

        # Use the provided logical vehicle name (e.g. "veh_lead")
        # directly as the topic namespace so it matches the bridge
        # and micro-ROS firmware topic scheme.
        self._name = name
        self._role = role
        self._slow_pid = slow_pid
        self._mid_pid = mid_pid
        self._fast_pid = fast_pid
        self._desired_time_headway = desired_time_headway
        self._min_spacing = min_spacing
        self._K_dist = K_dist

        # State coming from topics
        self._speed = 0.0          # current vehicle speed [m/s]
        self._setpoint = 0.0       # local ACC speed setpoint [m/s]
        self._dist_to_veh = 0.0    # distance to preceding vehicle [m]

        self._last_setpoint = 0.0  # Setpoint saving to avoid collisions when slowing down

        # Platoon state
        self._platoon_enabled = True # On by default for testing
        self._platoon_sp = 0.0

        # Debug / introspection variables (filled in compute_speed_setpoint)
        self._last_base_sp = 0.0
        self._last_eff_sp = 0.0
        self._last_dist_err = 0.0

        # Speed-range scheduling (matches Platooning.py thresholds)
        self._v1 = 11.11
        self._v2 = 22.22
        self._h = 1.0
        self._current_range = "low"  # "low" | "mid" | "high"

        self._dbg_print_idx = 0

        # Optional debug logging for timing/behavior (used mainly for leader)
        self._debug_step_idx = 0
        self._debug_writer = None
        if self._role == "lead":
            try:
                import csv  # local import to avoid unused at module level
                self._debug_log_file = open(f"{self._name}_debug.csv", mode="w", newline="")
                self._debug_writer = csv.writer(self._debug_log_file)
                self._debug_writer.writerow([
                    "step_idx",
                    "wall_time_s",
                    "speed_meas",
                    "speed_sp",
                    "dist_to_veh",
                    "platoon_sp",
                    "local_setpoint_base",
                ])
                self._debug_log_file.flush()
            except Exception:
                self._debug_writer = None

        # QoS profiles: BEST_EFFORT + VOLATILE everywhere (MCU compatibility).
        qos_state = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        qos_setpoint = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        qos_cmd = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # Subscriptions
        self._speed_sub = self.create_subscription(
            Float32,
            f"{self._name}/state/speed",
            self.speed_cb,
            qos_state,
        )

        # ACC / local setpoint
        self._setpoint_sub = self.create_subscription(
            Float32,
            f"{self._name}/state/setpoint",
            self.setpoint_cb,
            qos_setpoint,
        )

        # Distance to preceding vehicle
        self._dist_sub = self.create_subscription(
            Float32,
            f"{self._name}/state/dist_to_veh",
            self.dist_to_veh_cb,
            qos_state,
        )

        # Platoon coordination topics (same for all vehicles)
        self._platoon_sp_sub = self.create_subscription(
            Float32,
            f"/platoon/{platoon_id}/setpoint",
            self.platoon_sp_cb,
            qos_setpoint,
        )
        self._platoon_mode_sub = self.create_subscription(
            Bool,
            f"{self._name}/state/platoon_enabled",
            self.platoon_mode_cb,
            qos_setpoint,
        )

        # Publishers (unchanged)
        self._throttle_pub = self.create_publisher(
            Float32,
            f"{self._name}/command/throttle",
            qos_cmd,
        )
        self._brake_pub = self.create_publisher(
            Float32,
            f"{self._name}/command/brake",
            qos_cmd,
        )
        
        self._local_sp_pub = self.create_publisher(
            Float32,
            f"{self._name}/state/local_setpoint",
            qos_state,
        )
        
        # Timer-driven control loop by default. For fully deterministic
        # coupling to a simulator tick (e.g. CARLA 100 Hz), the caller can
        # disable this timer and invoke _control_loop or step_once() manually
        # once per simulation step.
        self._timer = self.create_timer(control_period, self._control_loop)

    # --- Callbacks for state topics ---

    def speed_cb(self, msg: Float32):
        self._speed = float(msg.data)

    def setpoint_cb(self, msg: Float32):
        self._setpoint = float(msg.data)

    def dist_to_veh_cb(self, msg: Float32):
        self._dist_to_veh = float(msg.data)

    def platoon_sp_cb(self, msg: Float32):
        self._last_setpoint = self._platoon_sp
        self._platoon_sp = float(msg.data)

    def platoon_mode_cb(self, msg: Bool):
        self._platoon_enabled = bool(msg.data)

    # --- Core control logic (CARLA-agnostic) ---

    def compute_speed_setpoint(self) -> float:
        """
        Compute the effective speed setpoint for this member.

        Exact behavioral port of Platooning.py:
          * lead: speed_sp = base_sp
          * follower: speed_sp = base_sp + d_sp, where
              desired_dist = max(min_spacing + 3.0, v*T + min_spacing)
              dist_err = dist_to_veh - desired_dist
              d_sp = K*dist_err if dist_err >= 0 else 2*K*dist_err

        base_sp selection:
          * platoon_enabled == False -> local setpoint (_setpoint)
          * platoon_enabled == True  -> global platoon setpoint (_platoon_sp)
        """
        speed = self._speed

        # Choose base setpoint
        if self._platoon_enabled:
            base_sp = self._platoon_sp
        else:
            base_sp = self._setpoint

        # Store for debugging
        self._last_base_sp = base_sp

        # Lead ignores distance logic (matches Platooning.py lead behavior).
        if self._role == "lead":
            self._last_dist_err = 0.0
            self._last_eff_sp = base_sp
            return base_sp

        # If we have no information about a vehicle ahead, use base setpoint.
        if self._dist_to_veh <= 0.0:
            self._last_dist_err = 0.0
            self._last_eff_sp = base_sp
            return base_sp

        desired_dist = max(
            self._min_spacing + 3.0,
            speed * self._desired_time_headway + self._min_spacing,
        )
        dist_err = self._dist_to_veh - desired_dist
        self._last_dist_err = dist_err

        if dist_err >= 0.0:
            d_sp = self._K_dist * dist_err
        else:
            d_sp = 2.0 * self._K_dist * dist_err

        eff_sp = base_sp + d_sp
        self._last_eff_sp = eff_sp
        return eff_sp

    def _select_pid(self) -> PID:
        if self._current_range == "high":
            return self._fast_pid
        if self._current_range == "mid":
            return self._mid_pid
        return self._slow_pid

    def _update_speed_range(self, speed: float) -> bool:
        prev = self._current_range

        if self._current_range == "low":
            if speed > self._v1 + self._h:
                self._current_range = "mid"
        elif self._current_range == "mid":
            if speed > self._v2 + self._h:
                self._current_range = "high"
            elif speed < self._v1 - self._h:
                self._current_range = "low"
        else:  # high
            if speed < self._v2 - self._h:
                self._current_range = "mid"

        return self._current_range != prev

    def _reset_selected_pid(self) -> None:
        try:
            self._select_pid().reset()
        except Exception:
            pass

    def compute_control(self):
        """
        Run PID on speed to get throttle/brake commands in [0, 1].
        Returns (throttle, brake).
        """
        speed_sp = self.compute_speed_setpoint()
        speed_meas = self._speed

        # Debug: for the leader, log each control step with wall time and
        # key signals so we can inspect effective timing and behavior.
        if self._debug_writer is not None:
            try:
                now = self.get_clock().now().to_msg()
                wall_t = float(now.sec) + float(now.nanosec) * 1e-9
                self._debug_writer.writerow([
                    self._debug_step_idx,
                    f"{wall_t:.9f}",
                    f"{speed_meas:.6f}",
                    f"{speed_sp:.6f}",
                    f"{self._dist_to_veh:.6f}",
                    f"{self._platoon_sp:.6f}",
                    f"{self._setpoint:.6f}",
                ])
                self._debug_log_file.flush()
                self._debug_step_idx += 1
            except Exception:
                pass

        if self._update_speed_range(speed_meas):
            self._reset_selected_pid()
        u = self._select_pid().step(speed_sp, speed_meas)

        if u > 0.0:
            throttle = float(u)
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(abs(u))

        # Lightweight periodic debug (helps diagnose "stuck at 0 throttle").
        self._dbg_print_idx += 1
        if (self._dbg_print_idx % 500) == 0 and self._role == "lead":
            try:
                self.get_logger().info(
                    f"[{self._name}] v={speed_meas:.2f} base_sp={self._last_base_sp:.2f} "
                    f"platoon_sp={self._platoon_sp:.2f} enabled={self._platoon_enabled} "
                    f"u={u:.3f} th={throttle:.3f} br={brake:.3f}"
                )
            except Exception:
                pass

        return throttle, brake, speed_sp

    def _control_loop(self):
        """
        Periodic timer callback: compute and publish throttle/brake commands.
        """
        throttle, brake, speed_sp = self.compute_control()

        self._throttle_pub.publish(Float32(data=throttle))
        self._brake_pub.publish(Float32(data=brake))
        self._local_sp_pub.publish(Float32(data=speed_sp))

        '''
        # Debug printout so we can see what each vehicle is doing.
        self.get_logger().info(
            f"[{self._name}] role={self._role} platoon={self._platoon_enabled} "
            f"v={self._speed:.2f} m/s dist={self._dist_to_veh:.2f} m "
            f"base_sp={self._last_base_sp:.2f} eff_sp={self._last_eff_sp:.2f} "
            f"throttle={throttle:.2f} brake={brake:.2f}"
        )
        '''

    def step_once(self) -> None:
        """Explicit one-shot control update.

        Call this exactly once per simulator tick when you want full
        control over timing instead of relying on the internal timer.
        """
        self._control_loop()
