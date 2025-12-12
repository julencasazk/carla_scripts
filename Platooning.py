import carla
import carla
import numpy as np
import math
from PID import PID

class PlatoonMember:
    def __init__(
        self,
        name: str,
        role: str,
        vehicle: carla.Vehicle,
        pid: PID,
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
        self.pid = pid

        # spacing policy params (only used for followers)
        self.desired_time_headway = desired_time_headway
        self.min_spacing = min_spacing
        self.K_dist = K_dist

        # will be filled each tick by platoon logic
        self.current_speed_sp = 0.0
        self.last_control = carla.VehicleControl(throttle=0.0, brake=0.0)

    def get_state(self):
        """Return scalar speed, accel magnitude, and transform."""
        vel = self.vehicle.get_velocity()
        accel = self.vehicle.get_acceleration()
        tf = self.vehicle.get_transform()

        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        accel_abs = math.sqrt(accel.x**2 + accel.y**2 + accel.z**2)
        return speed, accel_abs, tf

    def compute_control(
        self,
        dt: float,
        lead_state=None,
        global_lead_speed_sp: float = 0.0,
    ) -> carla.VehicleControl:
        """
        lead_state: (lead_speed, lead_tf) for followers
        global_lead_speed_sp: the high‑level speed setpoint used for the lead 
                              (and broadcast for CACC).

        Returns the control to be applied this tick.
        """
        speed, accel_abs, tf = self.get_state()

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
                lead_speed, lead_tf = lead_state

                # spacing policy
                # desired_dist = v * T + d0
                # TODO MUST Change later
                desired_dist = speed * self.desired_time_headway + self.min_spacing
                desired_dist = max(self.min_spacing + 3.0, desired_dist)

                rel_vect = lead_tf.location - tf.location
                ego_lead_dist = math.sqrt(
                    rel_vect.x**2 + rel_vect.y**2 + rel_vect.z**2
                )
                dist_err = ego_lead_dist - desired_dist

                if dist_err >= 0.0:
                    dSetpoint = self.K_dist * dist_err
                else:
                    # Just a heuristic, braking should be "twice" as
                    # important as accelerating
                    dSetpoint = 2.0 * self.K_dist * dist_err

                speed_sp = global_lead_speed_sp + dSetpoint
                self.current_speed_sp = speed_sp

        u = self.pid.step(speed_sp, speed)

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
        # Pre‑compute states
        states = []
        for m in self.members:
            speed, accel_abs, tf = m.get_state()
            states.append((speed, accel_abs, tf))

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
