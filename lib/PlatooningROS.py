"""
ROS2 platoon member node (CARLA-agnostic, MCU-like topic scheme).

Subscribes
  - /<name>/state/speed        (std_msgs/Float32)
  - /<name>/state/setpoint     (std_msgs/Float32)
  - /<name>/state/dist_to_veh  (std_msgs/Float32)
  - /<name>/state/desired_decel (std_msgs/Float32)

Publishes
  - /<name>/command/throttle   (std_msgs/Float32)  [0..1]
  - /<name>/command/brake      (std_msgs/Float32)  [0..1]
  - /<name>/state/desired_decel (std_msgs/Float32)

Behavior
  - Gain-scheduled speed PID, spacing policy, brake supervisor/mapping.

Notes
  - Simulator/vehicle specifics are handled in a separate bridge node.
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32, Bool

from lib.PID import PID


class PlatoonMember(Node):
    def __init__(
        self,
        name: str,
        platoon_index: int,
        slow_pid: PID,
        mid_pid: PID,
        fast_pid: PID,
        desired_time_headway: float = 0.5,
        min_spacing: float = 5.0,
        K_dist: float = 0.2,
        K_vel: float = 1.0,
        control_period: float = 0.05,
        platoon_id: str = "plat_0",
        K_brake = 1.0,
    ):
        """
        platoon_index: 0 for leader, 1..N-1 for followers.
        """
        super().__init__(name)

        # Use the provided logical vehicle name (e.g. "veh_0")
        # directly as the topic namespace so it matches the bridge
        # and micro-ROS firmware topic scheme.
        self._name = name
        self._platoon_index = int(platoon_index)
        self._slow_pid = slow_pid
        self._mid_pid = mid_pid
        self._fast_pid = fast_pid
        self._desired_time_headway = desired_time_headway
        self._min_spacing = min_spacing
        self._K_dist = K_dist
        self._K_vel = K_vel
        self._K_brake = K_brake

        self._control_period = float(control_period)

        # State coming from topics
        self._speed = 0.0          # current vehicle speed [m/s]
        self._setpoint = 0.0       # local ACC speed setpoint [m/s]
        self._dist_to_veh = 0.0    # distance to preceding vehicle [m]
        self._preceding_speed = 0.0 

        self._last_setpoint = 0.0  # Setpoint saving to avoid collisions when slowing down

        # Platoon state
        self._platoon_enabled = True # On by default for testing
        self._platoon_sp = 0.0

        # Debug / introspection variables (filled in compute_speed_setpoint)
        self._last_base_sp = 0.0
        self._last_eff_sp = 0.0
        self._last_dist_err = 0.0
        self._last_vel_diff = 0.0

        # Brake supervisor state/params
        self._brake_active = False
        self._brake_cmd_prev = 0.0

        # Throttle PID setpoint smoothing (prevents a throttle jump after braking)
        self._pid_speed_sp = 0.0
        self._pid_speed_sp_init = False
        # Limit how fast the PID setpoint can rise/fall [m/s^2]
        self._pid_sp_rate_up = 5.0
        self._pid_sp_rate_down = 10.0

        # Defaults chosen to be conservative and avoid chatter.
        self._brake_enable = True
        self._brake_deadband = 0.02
        self._brake_rate_limit = 1.0  # brake units per second

        # Hardcoded per-range brake calibration from CARLA test
        #
        # Mapping: desired_decel (m/s^2, positive magnitude) -> brake_cmd [0..1]
        # using linear interpolation between (min_decel -> min_brake)
        # and (max_decel -> max_brake).
        self._low_min_decel = 1.0
        self._low_max_decel = 5.0
        self._low_min_brake = 0.05
        self._low_max_brake = 0.50

        self._mid_min_decel = 1.0
        self._mid_max_decel = 6.0
        self._mid_min_brake = 0.05
        self._mid_max_brake = 0.50

        self._high_min_decel = 1.0
        self._high_max_decel = 8.0
        self._high_min_brake = 0.05
        self._high_max_brake = 0.50

        # Gates / hysteresis
        self._dist_on = 1.0
        self._dist_off = 0.5
        self._ttc_on = 2.5
        self._ttc_off = 3.2
        self._v_margin_on = 0.5
        self._v_margin_off = 0.2

        # Decel targets (magnitudes, m/s^2)
        self._decel_mild = 1.0
        self._decel_mod = 2.5
        self._decel_full = 5.0

        # Cooperative braking intent (desired decel magnitude, m/s^2)
        self._desired_decel = 0.0
        self._preceding_desired_decel = 0.0
        self._preceding_desired_decel_rx_t = None  # monotonic time when last message arrived
        self._coop_enable = True
        self._coop_timeout_s = 0.30
        self._coop_gain = 2.0
        self._coop_decel_on = 0.20
        self._coop_decel_off = 0.10

        # Speed-range scheduling (matches Platooning.py thresholds)
        self._v1 = 11.11
        self._v2 = 22.22
        self._h = 1.0
        self._current_range = "low"  # "low" | "mid" | "high"

        self._dbg_print_idx = 0

        # Optional debug logging for timing/behavior (used mainly for leader)
        self._debug_step_idx = 0
        self._debug_writer = None
        if self._platoon_index == 0:
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
            f"/{self._name}/state/speed",
            self.speed_cb,
            qos_state,
        )

        # ACC / local setpoint
        self._setpoint_sub = self.create_subscription(
            Float32,
            f"/{self._name}/state/setpoint",
            self.setpoint_cb,
            qos_setpoint,
        )

        # Distance to preceding vehicle
        self._dist_sub = self.create_subscription(
            Float32,
            f"/{self._name}/state/dist_to_veh",
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
            f"/{self._name}/state/platoon_enabled",
            self.platoon_mode_cb,
            qos_setpoint,
        )

        if self._platoon_index > 0: 
            self._preceding_speed_sub = self.create_subscription(
                Float32,
                f"/veh_{self._platoon_index - 1}/state/speed",
                self.preceding_speed_cb,
                qos_setpoint,
            )

            self._preceding_desired_decel_sub = self.create_subscription(
                Float32,
                f"/veh_{self._platoon_index - 1}/state/desired_decel",
                self.preceding_desired_decel_cb,
                qos_setpoint,
            )

        # Publishers
        self._throttle_pub = self.create_publisher(
            Float32,
            f"/{self._name}/command/throttle",
            qos_cmd,
        )
        self._brake_pub = self.create_publisher(
            Float32,
            f"/{self._name}/command/brake",
            qos_cmd,
        )

        self._local_sp_pub = self.create_publisher(
            Float32,
            f"/{self._name}/state/local_setpoint",
            qos_state,
        )

        self._desired_decel_pub = self.create_publisher(
            Float32,
            f"/{self._name}/state/desired_decel",
            qos_state,
        )


        # Timer-driven control loop by default. For fully deterministic
        # coupling to a simulator tick (e.g. CARLA 100 Hz), the caller can
        # disable this timer and invoke _control_loop or step_once() manually
        # once per simulation step.
        self._timer = self.create_timer(control_period, self._control_loop)

    def _decel_to_brake(self, speed_range: str, desired_decel: float) -> float:
        """Linear decel->brake mapping using hardcoded per-range constants."""
        if desired_decel <= 0.0:
            return 0.0

        if speed_range == "high":
            min_d, max_d = float(self._high_min_decel), float(self._high_max_decel)
            min_b, max_b = float(self._high_min_brake), float(self._high_max_brake)
        elif speed_range == "mid":
            min_d, max_d = float(self._mid_min_decel), float(self._mid_max_decel)
            min_b, max_b = float(self._mid_min_brake), float(self._mid_max_brake)
        else:
            min_d, max_d = float(self._low_min_decel), float(self._low_max_decel)
            min_b, max_b = float(self._low_min_brake), float(self._low_max_brake)

        if max_d <= min_d:
            return 0.0
        if max_b <= min_b:
            return max(0.0, min(1.0, min_b))

        if desired_decel <= min_d:
            cmd = min_b
        elif desired_decel >= max_d:
            cmd = max_b
        else:
            t = (desired_decel - min_d) / (max_d - min_d)
            cmd = min_b + t * (max_b - min_b)

        # Optional global gain
        cmd = float(self._K_brake) * float(cmd)
        return max(0.0, min(1.0, cmd))

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

    def preceding_speed_cb(self, msg: Float32):
        self._preceding_speed = float(msg.data)

    def preceding_desired_decel_cb(self, msg: Float32):
        # Cooperative braking intent is a positive decel magnitude [m/s^2].
        d = float(msg.data)
        self._preceding_desired_decel = max(0.0, d)
        self._preceding_desired_decel_rx_t = time.monotonic()

    def _fresh_preceding_desired_decel(self) -> float:
        if not self._coop_enable:
            return 0.0
        if self._platoon_index <= 0:
            return 0.0
        if self._preceding_desired_decel_rx_t is None:
            return 0.0
        age = time.monotonic() - float(self._preceding_desired_decel_rx_t)
        if age > float(self._coop_timeout_s):
            return 0.0
        return max(0.0, float(self._preceding_desired_decel))

    # --- Core control logic (CARLA-agnostic) ---

    def compute_speed_setpoint(self) -> float:
        """
        Compute the effective speed setpoint for this member.

        Exact behavioral port of Platooning.py:
          * leader (index 0): speed_sp = base_sp
          * follower (index > 0): speed_sp = base_sp + d_sp, where
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

        # Leader ignores distance logic (matches Platooning.py lead behavior).
        if self._platoon_index == 0:
            self._last_dist_err = 0.0
            self._last_eff_sp = base_sp
            return base_sp

        # If we have no information about a vehicle ahead, use base setpoint.
        if self._dist_to_veh <= 0.0:
            self._last_dist_err = 0.0
            self._last_eff_sp = base_sp
            return base_sp

        desired_dist, dist_err, vel_diff, _, _ = self._compute_spacing_signals()
        

        if dist_err >= 0.0:
            d_sp = self._K_dist * dist_err + self._K_vel * vel_diff
        else:
            d_sp = 4.0 * (self._K_dist * dist_err + self._K_vel * vel_diff)

        eff_sp = min(base_sp + d_sp, 33.33)
        eff_sp = max(eff_sp, 0.0)
        self._last_eff_sp = eff_sp
        return eff_sp

    def _compute_spacing_signals(self):
        """Compute spacing-related signals shared by setpoint and braking logic.

        Returns: (desired_dist, dist_err, vel_diff, closing_speed, ttc)
        - desired_dist: m
        - dist_err: d - desired_dist (m)
        - vel_diff: v_lead - v_ego (m/s)
        - closing_speed: max(v_ego - v_lead, 0) (m/s)
        - ttc: s (inf if not closing or invalid)
        """
        speed = float(self._speed)
        dist = float(self._dist_to_veh)

        desired_dist = max(
            float(self._min_spacing) + 3.0,
            speed * float(self._desired_time_headway) + float(self._min_spacing),
        )
        dist_err = dist - desired_dist

        # Lead speed can be missing early; treat as equal speed (no closing).
        v_lead = float(self._preceding_speed) if self._preceding_speed is not None else speed
        vel_diff = v_lead - speed
        closing_speed = max(speed - v_lead, 0.0)
        if dist > 0.0 and closing_speed > 0.1:
            ttc = dist / closing_speed
        else:
            ttc = float("inf")

        self._last_dist_err = float(dist_err)
        self._last_vel_diff = float(vel_diff)
        return float(desired_dist), float(dist_err), float(vel_diff), float(closing_speed), float(ttc)

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

        # NOTE: For step-response testing, feed the raw effective setpoint
        # directly to the speed PID (no internal setpoint ramping).

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

        # Throttle-only speed PID (no direct braking from negative u)
        u = float(self._select_pid().step(float(speed_sp), speed_meas))
        throttle_cmd = max(0.0, min(1.0, u))

        brake_cmd = float(self.compute_brake(speed_sp=speed_sp, throttle_req=throttle_cmd))
        if brake_cmd > 0.0:
            throttle_cmd = 0.0


        # Lightweight periodic debug (helps diagnose "stuck at 0 throttle").
        self._dbg_print_idx += 1
        if (self._dbg_print_idx % 500) == 0 and self._platoon_index == 0:
            try:
                self.get_logger().info(
                    f"[{self._name}] v={speed_meas:.2f} base_sp={self._last_base_sp:.2f} "
                    f"platoon_sp={self._platoon_sp:.2f} enabled={self._platoon_enabled} "
                    f"th={throttle_cmd:.3f} br={brake_cmd:.3f}"
                )
            except Exception:
                pass

        # Publish the effective (unramped) setpoint.
        return throttle_cmd, brake_cmd, float(speed_sp)
    
    def compute_brake(self, speed_sp: float, throttle_req: float) -> float:
        """Brake supervisor.

        - Uses distance/TTC/overspeed gates to decide when braking is needed.
        - Uses a hardcoded per-range decel->brake linear mapping.
        - Applies hysteresis and rate limiting to avoid chatter.
        """
        if not self._brake_enable:
            return 0.0

        v = float(self._speed)
        d = float(self._dist_to_veh)
        desired_dist, dist_err, vel_diff, closing_speed, ttc = self._compute_spacing_signals()

        # Determine speed bin using the same range scheduler.
        bin_name = self._current_range

        # --- Gate evaluation ---
        too_close_on = (self._platoon_index > 0) and (d > 0.0) and (dist_err < -float(self._dist_on))
        too_close_off = (self._platoon_index > 0) and (d > 0.0) and (dist_err < -float(self._dist_off))

        ttc_on = (self._platoon_index > 0) and (ttc < float(self._ttc_on))
        ttc_off = (self._platoon_index > 0) and (ttc < float(self._ttc_off))

        # Overspeed brake gate: only if we're above speed_sp and already asking for near-zero throttle.
        overspeed = v - float(speed_sp)
        overspeed_on = (overspeed > float(self._v_margin_on)) and (throttle_req <= 0.05)
        overspeed_off = (overspeed > float(self._v_margin_off)) and (throttle_req <= 0.05)

        # Cooperative braking feedforward (from predecessor), with receiver-side freshness.
        ff_decel = float(self._fresh_preceding_desired_decel())
        ff_on = ff_decel >= float(self._coop_decel_on)
        ff_off = ff_decel >= float(self._coop_decel_off)

        if not self._brake_active:
            want_brake = bool(too_close_on or ttc_on or overspeed_on or ff_on)
        else:
            want_brake = bool(too_close_off or ttc_off or overspeed_off or ff_off)

        # If braking is not desired, ramp down with rate limit.
        if not want_brake:
            self._brake_active = False
            target = 0.0
            self._desired_decel = 0.0
        else:
            if not self._brake_active:
                self._brake_active = True
                # Reset PID integrator when we start braking to avoid a throttle kick when releasing.
                self._reset_selected_pid()

            # Local 3-level severity (feedback):
            # - mild: overspeed only
            # - moderate: normal closing / too-close gate
            # - full: urgent (very low TTC or very negative distance error)
            if too_close_on or ttc_on:
                # Upgrade to full only when it's clearly urgent.
                if (ttc < 1.2) or (dist_err < -2.0):
                    fb_decel = float(self._decel_full)
                else:
                    fb_decel = float(self._decel_mod)
            elif overspeed_on or overspeed_off:
                fb_decel = float(self._decel_mild)
            else:
                fb_decel = 0.0

            # Cooperative braking arbitration (feedforward can only add braking).
            desired_decel = max(float(fb_decel), float(self._coop_gain) * float(ff_decel))
            self._desired_decel = float(desired_decel)

            target = self._decel_to_brake(speed_range=bin_name, desired_decel=desired_decel)

        # Rate limit
        max_step = float(self._brake_rate_limit) * max(self._control_period, 1e-3)
        lo = self._brake_cmd_prev - max_step
        hi = self._brake_cmd_prev + max_step
        cmd = max(lo, min(hi, target))

        # Deadband near zero
        if cmd < float(self._brake_deadband) and not self._brake_active:
            cmd = 0.0

        cmd = max(0.0, min(1.0, float(cmd)))
        self._brake_cmd_prev = cmd
        return cmd

    def _control_loop(self):
        """
        Periodic timer callback: compute and publish throttle/brake commands.
        """
        throttle, brake, speed_sp = self.compute_control()

        self._throttle_pub.publish(Float32(data=throttle))
        self._brake_pub.publish(Float32(data=brake))
        self._local_sp_pub.publish(Float32(data=speed_sp))
        self._desired_decel_pub.publish(Float32(data=self._desired_decel))

        '''
        # Debug printout so we can see what each vehicle is doing.
        self.get_logger().info(
            f"[{self._name}] idx={self._platoon_index} platoon={self._platoon_enabled} "
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
