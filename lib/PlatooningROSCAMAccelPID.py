"""
ROS2 platoon member node (CARLA-agnostic) with cascaded control:
outer speed PID -> desired accel -> inner accel PID -> throttle/brake.

Subscribes
  - /<name>/state              (platooning_msgs/VehicleState)
  - /<name>/state/setpoint     (std_msgs/Float32)
  - /platoon/<id>/setpoint     (std_msgs/Float32)
  - /veh_<idx>/cam             (etsi_its_lite_msgs/CAM)

Publishes
  - /<name>/command/throttle   (std_msgs/Float32)
  - /<name>/command/brake      (std_msgs/Float32)
  - /<name>/state/local_setpoint (std_msgs/Float32)
  - /veh_<idx>/cam             (etsi_its_lite_msgs/CAM)

Behavior
  - Gain-scheduled outer speed PID and inner accel PID(s).
  - Optional panic gains, deadbands, cooperative braking intent.

Notes
  - Simulator/vehicle specifics are handled in a separate bridge node.
"""

import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32, Bool
from platooning_msgs.msg import VehicleState
from etsi_its_lite_msgs.msg import CAM

from lib.PID import PID


class PlatoonMember(Node):
    def __init__(
        self,
        name: str,
        platoon_index: int,
        speed_pid_low: PID,
        speed_pid_mid: PID,
        speed_pid_high: PID,
        accel_pid_low: PID,
        accel_pid_mid: PID,
        accel_pid_high: PID,
        desired_time_headway: float = 0.5,
        min_spacing: float = 5.0,
        K_dist: float = 0.2,
        K_vel: float = 1.0,
        K_dist_panic: Optional[float] = None,
        K_vel_panic: Optional[float] = None,
        ttc_panic_s: float = 2.0,
        dist_err_deadband_m: float = 0.0,
        vel_diff_deadband_mps: float = 0.0,
        control_period: float = 0.05,
        platoon_id: str = "plat_0",
    ):
        super().__init__(name)

        self._name = name
        self._platoon_index = int(platoon_index)
        self._speed_pid_low = speed_pid_low
        self._speed_pid_mid = speed_pid_mid
        self._speed_pid_high = speed_pid_high
        self._accel_pid_low = accel_pid_low
        self._accel_pid_mid = accel_pid_mid
        self._accel_pid_high = accel_pid_high

        self._desired_time_headway = float(desired_time_headway)
        self._min_spacing = float(min_spacing)
        self._K_dist_normal = float(K_dist)
        self._K_vel_normal = float(K_vel)
        self._K_dist_panic = float(K_dist_panic) if K_dist_panic is not None else float(K_dist)
        self._K_vel_panic = float(K_vel_panic) if K_vel_panic is not None else float(K_vel)
        self._ttc_panic_s = float(ttc_panic_s)
        self._dist_err_deadband_m = float(dist_err_deadband_m)
        self._vel_diff_deadband_mps = float(vel_diff_deadband_mps)
        self._control_period = float(control_period)
        self._platoon_id = str(platoon_id)

        # State from topics
        self._speed = 0.0
        self._accel_long_mps2 = 0.0
        self._setpoint = 0.0
        self._dist_to_veh = 0.0
        self._preceding_speed = 0.0

        # Platoon state
        self._platoon_enabled = True
        self._platoon_sp = 0.0

        # Range scheduling
        self._v1 = 11.11
        self._v2 = 22.22
        self._h = 1.0
        self._current_range = "low"  # "low" | "mid" | "high"

        # Local setpoint debug signals
        self._last_base_sp = 0.0
        self._last_eff_sp = 0.0
        self._last_dist_err = 0.0
        self._last_vel_diff = 0.0

        # Simple actuator arbitration hysteresis to avoid chatter near 0 accel.
        self._mode = "coast"  # "throttle" | "brake" | "coast"
        self._accel_deadband_on = 0.10
        self._accel_deadband_off = 0.05

        # Console debug prints (throttle/brake/speed/sp/accel)
        self._dbg_print_idx = 0
        self._dbg_print_every = 10  # control ticks; set to 1 for every tick

        # Safety / cooperative braking intent (optional, mirrors PlatooningROSCAM.py)
        self._desired_decel = 0.0  # magnitude m/s^2
        self._preceding_desired_decel = 0.0
        self._preceding_desired_decel_rx_t = None
        self._coop_enable = True
        self._coop_timeout_s = 0.30
        self._coop_gain = 1.0
        self._coop_decel_on = 0.20
        self._coop_decel_off = 0.10

        # Simple gates to request braking regardless of outer loop.
        self._brake_enable = True
        self._dist_on = 1.0
        self._dist_off = 0.5
        self._ttc_on = 2.5
        self._ttc_off = 3.2
        self._v_margin_on = 0.5
        self._v_margin_off = 0.2
        self._brake_active = False

        # CAM identity (same scheme as PlatooningROSCAM.py)
        self._station_id = int(self._platoon_index)
        self._preceding_station_id = self._station_id - 1 if self._platoon_index > 0 else None
        self._station_type = 5  # passengerCar
        self._cam_topic = f"/veh_{self._station_id}/cam"
        self._preceding_cam_topic = (
            f"/veh_{self._preceding_station_id}/cam" if self._preceding_station_id is not None else None
        )

        self._pos_x_m = 0.0
        self._pos_y_m = 0.0
        self._pos_z_m = 0.0
        self._last_state_stamp = None

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
        qos_cam = QoSProfile(
            depth=20,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # Subscriptions
        self._state_sub = self.create_subscription(
            VehicleState,
            f"/{self._name}/state",
            self.state_cb,
            qos_state,
        )
        self._setpoint_sub = self.create_subscription(
            Float32,
            f"/{self._name}/state/setpoint",
            self.setpoint_cb,
            qos_setpoint,
        )
        self._platoon_sp_sub = self.create_subscription(
            Float32,
            f"/platoon/{self._platoon_id}/setpoint",
            self.platoon_sp_cb,
            qos_setpoint,
        )

        if self._preceding_cam_topic is not None:
            self._preceding_cam_sub = self.create_subscription(
                CAM, self._preceding_cam_topic, self.preceding_cam_cb, qos_cam
            )
        else:
            self._preceding_cam_sub = None

        # Publishers
        self._throttle_pub = self.create_publisher(
            Float32, f"/{self._name}/command/throttle", qos_cmd
        )
        self._brake_pub = self.create_publisher(
            Float32, f"/{self._name}/command/brake", qos_cmd
        )
        self._local_sp_pub = self.create_publisher(
            Float32, f"/{self._name}/state/local_setpoint", qos_setpoint
        )
        self._cam_pub = self.create_publisher(CAM, self._cam_topic, qos_cam)

        # Debug publishers for test harnesses (best-effort)
        self._a_des_pub = self.create_publisher(Float32, f"/{self._name}/debug/a_des", qos_cmd)
        self._a_err_pub = self.create_publisher(Float32, f"/{self._name}/debug/a_err", qos_cmd)
        self._v_err_pub = self.create_publisher(Float32, f"/{self._name}/debug/v_err", qos_cmd)

        self._timer = self.create_timer(self._control_period, self._control_loop)

    def _set_pid_gains(self, pid: PID, kp: float, ki: float, kd: float, n: float) -> None:
        try:
            pid.kp = float(kp)
            pid.ki = float(ki)
            pid.kd = float(kd)
            pid.n = float(n)
            try:
                pid._update_derivative_coeffs()
            except Exception:
                pass
            pid.reset()
        except Exception:
            pass

    def set_gain_schedule(self, speed_gains, accel_gains) -> None:
        """Update PID gains (all ranges) and reset states.

        speed_gains/accel_gains: (kp, ki, kd, N)
        """
        skp, ski, skd, sN = speed_gains
        akp, aki, akd, aN = accel_gains
        for pid in (self._speed_pid_low, self._speed_pid_mid, self._speed_pid_high):
            self._set_pid_gains(pid, skp, ski, skd, sN)
        for pid in (self._accel_pid_low, self._accel_pid_mid, self._accel_pid_high):
            self._set_pid_gains(pid, akp, aki, akd, aN)

    # ----------------------------
    # Topic callbacks
    # ----------------------------
    def state_cb(self, msg: VehicleState):
        self._speed = float(msg.speed_mps)
        self._accel_long_mps2 = float(msg.accel_mps2)
        self._dist_to_veh = float(msg.dist_to_prev)
        self._pos_x_m = float(msg.pos_x_m)
        self._pos_y_m = float(msg.pos_y_m)
        self._pos_z_m = float(msg.pos_z_m)
        self._last_state_stamp = msg.timestamp

    def setpoint_cb(self, msg: Float32):
        self._setpoint = float(msg.data)

    def platoon_sp_cb(self, msg: Float32):
        self._platoon_sp = float(msg.data)

    def preceding_cam_cb(self, msg: CAM):
        # Preceding vehicle speed and decel intent
        try:
            speed_raw = int(msg.cam.cam_parameters.high_frequency_container.speed)
            self._preceding_speed = float(speed_raw) / 100.0
        except Exception:
            pass
        try:
            decel_raw = int(msg.cam.cam_parameters.high_frequency_container.deceleration_intent)
            self._preceding_desired_decel = float(decel_raw) / 10.0
            self._preceding_desired_decel_rx_t = time.monotonic()
        except Exception:
            pass

    # ----------------------------
    # Setpoint computation (same structure as PlatooningROSCAM.py)
    # ----------------------------
    def _compute_spacing_signals(self):
        speed = float(self._speed)
        dist = float(self._dist_to_veh)
        desired_dist = max(
            float(self._min_spacing) + 3.0,
            speed * float(self._desired_time_headway) + float(self._min_spacing),
        )

        v_lead = float(self._preceding_speed) if self._preceding_speed is not None else speed
        vel_diff = v_lead - speed
        closing_speed = max(speed - v_lead, 0.0)
        if dist > 0.0 and closing_speed > 0.1:
            ttc = dist / closing_speed
        else:
            ttc = float("inf")
        dist_err = dist - desired_dist
        return float(desired_dist), float(dist_err), float(vel_diff), float(closing_speed), float(ttc)

    def compute_speed_setpoint(self) -> float:
        speed = float(self._speed)

        base_sp = float(self._platoon_sp) if self._platoon_enabled else float(self._setpoint)
        self._last_base_sp = float(base_sp)

        if self._platoon_index == 0:
            self._last_eff_sp = float(base_sp)
            self._last_dist_err = 0.0
            self._last_vel_diff = 0.0
            return float(base_sp)

        desired_dist, dist_err, vel_diff, _, ttc = self._compute_spacing_signals()
        self._last_dist_err = float(dist_err)
        self._last_vel_diff = float(vel_diff)

        in_panic = bool(float(ttc) <= float(self._ttc_panic_s))
        K_dist = float(self._K_dist_panic if in_panic else self._K_dist_normal)
        K_vel = float(self._K_vel_panic if in_panic else self._K_vel_normal)

        if abs(float(dist_err)) < float(self._dist_err_deadband_m):
            dist_err = 0.0
        if abs(float(vel_diff)) < float(self._vel_diff_deadband_mps):
            vel_diff = 0.0

        d_sp = (K_dist * float(dist_err)) + (K_vel * float(vel_diff))
        eff_sp = float(base_sp + d_sp)
        self._last_eff_sp = float(eff_sp)
        return float(max(0.0, eff_sp))

    # ----------------------------
    # Control logic
    # ----------------------------
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
        else:
            if speed < self._v2 - self._h:
                self._current_range = "mid"
        return self._current_range != prev

    def _select_speed_pid(self) -> PID:
        if self._current_range == "high":
            return self._speed_pid_high
        if self._current_range == "mid":
            return self._speed_pid_mid
        return self._speed_pid_low

    def _select_accel_pid(self) -> PID:
        if self._current_range == "high":
            return self._accel_pid_high
        if self._current_range == "mid":
            return self._accel_pid_mid
        return self._accel_pid_low

    def _reset_pids(self) -> None:
        for pid in [
            self._speed_pid_low,
            self._speed_pid_mid,
            self._speed_pid_high,
            self._accel_pid_low,
            self._accel_pid_mid,
            self._accel_pid_high,
        ]:
            try:
                pid.reset()
            except Exception:
                pass

    def _fresh_preceding_desired_decel(self) -> float:
        if not self._coop_enable:
            return 0.0
        if self._preceding_desired_decel_rx_t is None:
            return 0.0
        age = time.monotonic() - float(self._preceding_desired_decel_rx_t)
        if age > float(self._coop_timeout_s):
            return 0.0
        return float(self._preceding_desired_decel)

    def _compute_safety_decel(self, speed_sp: float, throttle_req: float) -> float:
        if not self._brake_enable:
            return 0.0

        v = float(self._speed)
        d = float(self._dist_to_veh)
        desired_dist, dist_err, vel_diff, closing_speed, ttc = self._compute_spacing_signals()

        too_close_on = (self._platoon_index > 0) and (d > 0.0) and (dist_err < -float(self._dist_on))
        too_close_off = (self._platoon_index > 0) and (d > 0.0) and (dist_err < -float(self._dist_off))

        ttc_on = (self._platoon_index > 0) and (ttc < float(self._ttc_on))
        ttc_off = (self._platoon_index > 0) and (ttc < float(self._ttc_off))

        overspeed = v - float(speed_sp)
        overspeed_on = (overspeed > float(self._v_margin_on)) and (throttle_req <= 0.05)
        overspeed_off = (overspeed > float(self._v_margin_off)) and (throttle_req <= 0.05)

        ff_decel = float(self._fresh_preceding_desired_decel())
        ff_on = ff_decel >= float(self._coop_decel_on)
        ff_off = ff_decel >= float(self._coop_decel_off)

        if not self._brake_active:
            want = bool(too_close_on or ttc_on or overspeed_on or ff_on)
        else:
            want = bool(too_close_off or ttc_off or overspeed_off or ff_off)

        if not want:
            self._brake_active = False
            self._desired_decel = 0.0
            return 0.0

        self._brake_active = True
        # Simple severity schedule (placeholder)
        fb = 0.0
        if too_close_on or ttc_on:
            fb = 4.0 if (ttc < 1.2 or dist_err < -2.0) else 2.0
        elif overspeed_on or overspeed_off:
            fb = 1.0
        desired = max(float(fb), float(self._coop_gain) * float(ff_decel))
        self._desired_decel = float(desired)
        return float(desired)

    def compute_control(self):
        speed_sp = float(self.compute_speed_setpoint())
        speed_meas = float(self._speed)

        if self._update_speed_range(speed_meas):
            self._reset_pids()

        # Outer PID: speed -> desired accel (m/s^2)
        a_des = float(self._select_speed_pid().step(speed_sp, speed_meas))

        accel_pid = self._select_accel_pid()

        # Pre-arbitration throttle request estimate for overspeed gating in safety supervisor.
        # Avoid stepping the accel PID twice per cycle (integrator/derivative would advance twice).
        a_meas = float(self._accel_long_mps2)
        throttle_req = 1.0 if a_des > float(self._accel_deadband_on) else 0.0

        # Safety supervisor can request additional braking via a negative accel override.
        safety_decel = float(self._compute_safety_decel(speed_sp=speed_sp, throttle_req=throttle_req))
        if safety_decel > 0.0:
            a_des = min(a_des, -safety_decel)

        # Inner PID (single): desired accel -> signed effort
        u_effort = float(accel_pid.step(a_des, a_meas))
        u_effort = float(max(-1.0, min(1.0, u_effort)))

        # Mode hysteresis around zero to reduce chatter.
        if self._mode == "coast":
            if u_effort > float(self._accel_deadband_on):
                self._mode = "throttle"
            elif u_effort < -float(self._accel_deadband_on):
                self._mode = "brake"
        elif self._mode == "throttle":
            if u_effort < float(self._accel_deadband_off):
                self._mode = "coast"
        else:  # brake
            if u_effort > -float(self._accel_deadband_off):
                self._mode = "coast"

        throttle_cmd = 0.0
        brake_cmd = 0.0
        if self._mode == "throttle":
            throttle_cmd = max(0.0, min(1.0, u_effort))
        elif self._mode == "brake":
            brake_cmd = max(0.0, min(1.0, -u_effort))

        return float(throttle_cmd), float(brake_cmd), float(speed_sp), float(a_des)

    # ----------------------------
    # Periodic loop + CAM publish
    # ----------------------------
    def _control_loop(self):
        throttle, brake, speed_sp, a_des = self.compute_control()

        self._throttle_pub.publish(Float32(data=float(throttle)))
        self._brake_pub.publish(Float32(data=float(brake)))
        self._local_sp_pub.publish(Float32(data=float(speed_sp)))

        try:
            self._a_des_pub.publish(Float32(data=float(a_des)))
            self._a_err_pub.publish(Float32(data=float(a_des) - float(self._accel_long_mps2)))
            self._v_err_pub.publish(Float32(data=float(speed_sp) - float(self._speed)))
        except Exception:
            pass

        self._dbg_print_idx += 1
        if (self._dbg_print_every > 0) and ((self._dbg_print_idx % self._dbg_print_every) == 0):
            try:
                self.get_logger().info(
                    f"[{self._name}] v={self._speed:.2f} sp={float(speed_sp):.2f} "
                    f"a_des={float(a_des):.2f} a_meas={self._accel_long_mps2:.2f} "
                    f"th={float(throttle):.3f} br={float(brake):.3f} range={self._current_range} mode={self._mode}"
                )
            except Exception:
                pass

        # Publish CAM at a reduced rate (simple gating).
        now_sim_s = None
        if self._last_state_stamp is not None:
            now_sim_s = float(self._last_state_stamp.sec) + float(self._last_state_stamp.nanosec) * 1e-9
        if now_sim_s is None:
            now_sim_s = time.time()

        cam_msg = CAM()
        cam_msg.header.protocol_version = 1
        cam_msg.header.message_id = 2
        cam_msg.header.station_id = int(self._station_id)
        cam_msg.cam.timestamp = int((now_sim_s * 1000.0) % 65536)
        cam_msg.cam.cam_parameters.basic_container.station_type = int(self._station_type)
        # etsi_its_lite_msgs uses uint32 arrays for reference_position; CARLA world coords can be negative.
        # Encode as unsigned 32-bit to avoid OverflowError on negative values.
        def _u32_from_meters(m: float) -> int:
            cm = int(round(float(m) * 100.0))
            return int(cm) & 0xFFFFFFFF

        cam_msg.cam.cam_parameters.basic_container.reference_position = [
            _u32_from_meters(self._pos_x_m),
            _u32_from_meters(self._pos_y_m),
            _u32_from_meters(self._pos_z_m),
        ]
        cam_msg.cam.cam_parameters.high_frequency_container.speed = int(max(0, min(65535, round(self._speed * 100.0))))
        cam_msg.cam.cam_parameters.high_frequency_container.longitudinal_acceleration = int(
            max(0, min(255, round(abs(self._accel_long_mps2) * 10.0)))
        )
        cam_msg.cam.cam_parameters.high_frequency_container.aceleration_control = int(
            (1 if brake > 0.01 else 0) | (2 if throttle > 0.01 else 0)
        )
        cam_decel = max(0.0, -float(a_des))
        cam_msg.cam.cam_parameters.high_frequency_container.deceleration_intent = int(
            max(0, min(255, round(cam_decel * 10.0)))
        )
        cam_msg.cam.cam_parameters.high_frequency_container.platoon_position = int(self._platoon_index)
        cam_msg.cam.cam_parameters.high_frequency_container.platoon_id = 0
        self._cam_pub.publish(cam_msg)

    def step_once(self) -> None:
        self._control_loop()
