"""
CARLA-only platoon logic (no ROS). Defines Platoon and PlatoonMember helpers
for lead/follower roles, spacing policy, and gain-scheduled speed PID control.

Usage
  - Import from CARLA scripts that run without ROS.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

try:
    import carla  # type: ignore
except ModuleNotFoundError:  # Allows importing this module without CARLA installed.
    carla = None  # type: ignore

from lib.PID import PID

SpeedRange = Literal["low", "mid", "high"]


class PlatoonMember:
    def __init__(
        self,
        name: str,
        role: str,
        vehicle: "carla.Vehicle",
        pid: PID = None,
        desired_time_headway: float = 0.5,
        min_spacing: float = 5.0,
        K_dist: float = 0.2,
    ):
        """
        role: "lead" or "follower"
        For follower, we use a distance-based correction to the speed setpoint.
        """
        self.name = name
        self.role = role
        self.vehicle = vehicle

        # PID controller parameters (matched from reference test script)
        Ts = 0.01
        u_min = 0.0
        u_max = 1.0
        kb_aw = 1.0

        # Create separate PIDs for low/mid/high speed ranges. If a single
        # `pid` instance is provided, use it for all ranges to preserve
        # external configuration.
        if pid is None:
            self.pid_low = PID(0.43127789, 0.43676547, 0.0, 15.0, Ts, u_min, u_max, kb_aw, der_on_meas=True)
            self.pid_mid = PID(0.11675119, 0.085938,   0.0, 14.90530836, Ts, u_min, u_max, kb_aw, der_on_meas=True)
            self.pid_high = PID(0.13408096, 0.07281374,  0.0, 12.16810135, Ts, u_min, u_max, kb_aw, der_on_meas=True)
        else:
            # reuse provided pid for all ranges
            self.pid_low = pid
            self.pid_mid = pid
            self.pid_high = pid

        # Range thresholds (m/s) and hysteresis
        self.v1 = 11.11
        self.v2 = 22.22
        self.h = 1.0
        self.current_range: SpeedRange = "low"
        self.range_id_map = {"low": 0, "mid": 1, "high": 2}

        # spacing policy params (only used for followers)
        self.desired_time_headway = desired_time_headway
        self.min_spacing = min_spacing
        self.K_dist = K_dist

        # will be filled each tick by platoon logic
        self.current_speed_sp = 0.0
        self.last_control: Optional["carla.VehicleControl"] = None

    def get_state(self):
        """Return scalar speed, accel magnitude, and transform."""
        if carla is None:
            raise RuntimeError("CARLA Python API (module `carla`) is required to use PlatoonMember.")
        vel = self.vehicle.get_velocity()
        accel = self.vehicle.get_acceleration()
        tf = self.vehicle.get_transform()

        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        accel_abs = math.sqrt(accel.x**2 + accel.y**2 + accel.z**2)
        return speed, accel_abs, tf

    def _desired_follow_distance(self, ego_speed: float) -> float:
        # desired_dist = v * T + d0
        desired_dist = ego_speed * self.desired_time_headway + self.min_spacing
        # Legacy extra buffer kept as-is for backwards behavior.
        return max(self.min_spacing + 3.0, desired_dist)

    def _speed_setpoint_offset_from_distance_error(self, dist_err: float) -> float:
        if dist_err >= 0.0:
            return self.K_dist * dist_err
        # Just a heuristic, braking should be "twice" as important as accelerating
        return 2.0 * self.K_dist * dist_err

    def _update_speed_range(self, speed: float) -> bool:
        prev_range = self.current_range

        if self.current_range == "low":
            if speed > self.v1 + self.h:
                self.current_range = "mid"
        elif self.current_range == "mid":
            if speed > self.v2 + self.h:
                self.current_range = "high"
            elif speed < self.v1 - self.h:
                self.current_range = "low"
        else:  # high
            if speed < self.v2 - self.h:
                self.current_range = "mid"

        return self.current_range != prev_range

    def _reset_selected_pid(self) -> None:
        pid = self._select_pid()
        try:
            pid.reset()
        except Exception:
            pass

    def _select_pid(self) -> PID:
        if self.current_range == "high":
            return self.pid_high
        if self.current_range == "mid":
            return self.pid_mid
        return self.pid_low

    def compute_control(
        self,
        dt: float,
        lead_state=None,
        global_lead_speed_sp: float = 0.0,
    ) -> "carla.VehicleControl":
        """
        lead_state: (lead_speed, lead_tf) for followers
        global_lead_speed_sp: the high-level speed setpoint used for the lead 
                              (and broadcast for CACC).

        Returns the control to be applied this tick.
        """
        if carla is None:
            raise RuntimeError("CARLA Python API (module `carla`) is required to compute controls.")
        speed, _accel_abs, tf = self.get_state()

        if self.role == "lead":
            # Lead follows standard ACC for now
            speed_sp = global_lead_speed_sp
            self.current_speed_sp = speed_sp

        else:  # follower
            if lead_state is None:
                # fall back to keeping own speed if no info (should not happen in a proper platoon)
                speed_sp = speed
                self.current_speed_sp = speed_sp
            else:
                _lead_speed, lead_tf = lead_state

                # spacing policy
                desired_dist = self._desired_follow_distance(speed)

                rel_vect = lead_tf.location - tf.location
                ego_lead_dist = math.sqrt(
                    rel_vect.x**2 + rel_vect.y**2 + rel_vect.z**2
                )
                dist_err = ego_lead_dist - desired_dist

                speed_sp = global_lead_speed_sp + self._speed_setpoint_offset_from_distance_error(dist_err)
                self.current_speed_sp = speed_sp

        # Select PID controller based on current speed range and apply
        if self._update_speed_range(speed):
            # reset the newly selected PID to avoid integrator wind-up / transients
            self._reset_selected_pid()

        u = self._select_pid().step(speed_sp, speed)

        if u > 0.0:
            throttle_cmd = u
            brake_cmd = 0.0
        else:
            throttle_cmd = 0.0
            brake_cmd = abs(u)

        ctrl = carla.VehicleControl(
            throttle=float(throttle_cmd),
            brake=float(brake_cmd),
            steer=0.0,
            hand_brake=False,
            reverse=False,
        )
        self.last_control = ctrl
        return ctrl


class Platoon:
    def __init__(self, members):
        """
        members: list[PlatoonMember] in order [lead, follower1, follower2, ...]
        """
        self.members = members

    @property
    def lead(self) -> PlatoonMember:
        return self.members[0]

    def update(self, dt: float, global_lead_speed_sp: float):
        """
        One simulation step: compute and apply control for all members.
        For followers we pass their immediate predecessor as 'lead_state'.
        """
        # Pre-compute states
        states = [m.get_state() for m in self.members]

        # Controls for each
        controls = []

        # Lead
        lead_ctrl = self.lead.compute_control(
            dt=dt,
            lead_state=None,
            global_lead_speed_sp=global_lead_speed_sp,
        )
        controls.append(lead_ctrl)

        # Followers
        for i in range(1, len(self.members)):
            m = self.members[i]
            lead_speed, _, lead_tf = states[i - 1]
            follower_ctrl = m.compute_control(
                dt=dt,
                lead_state=(lead_speed, lead_tf),
                global_lead_speed_sp=global_lead_speed_sp,
            )
            controls.append(follower_ctrl)

        # Apply controls
        for m, ctrl in zip(self.members, controls):
            m.vehicle.apply_control(ctrl)

        return states, controls
