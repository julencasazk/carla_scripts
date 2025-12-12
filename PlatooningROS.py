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
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32, Bool

from PID import PID


class PlatoonMember(Node):
    def __init__(
        self,
        name: str,
        role: str,
        pid: PID,
        desired_time_headway: float = 0.5,
        min_spacing: float = 5.0,
        K_dist: float = 0.2,
        control_period: float = 0.05,
        platoon_id: str = "main",
    ):
        """
        role: "lead" or "follower"
        For follower, a distance-based correction is applied to the speed setpoint.
        """
        super().__init__(name)

        self._name = f"{name}_node"
        self._role = role
        self._pid = pid
        self._desired_time_headway = desired_time_headway
        self._min_spacing = min_spacing
        self._K_dist = K_dist

        # State coming from topics
        self._speed = 0.0          # current vehicle speed [m/s]
        self._setpoint = 0.0       # local ACC speed setpoint [m/s]
        self._dist_to_veh = 0.0    # distance to preceding vehicle [m]

        # Platoon state
        self._platoon_enabled = False
        self._platoon_sp = 0.0

        qos_subs = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        qos_pubs = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )

        # Subscriptions
        self._speed_sub = self.create_subscription(
            Float32,
            f"{self._name}/state/speed",
            self.speed_cb,
            qos_pubs,
        )

        # ACC / local setpoint
        self._setpoint_sub = self.create_subscription(
            Float32,
            f"{self._name}/state/setpoint",
            self.setpoint_cb,
            qos_pubs,
        )

        # Distance to preceding vehicle
        self._dist_sub = self.create_subscription(
            Float32,
            f"{self._name}/state/dist_to_veh",
            self.dist_to_veh_cb,
            qos_pubs,
        )

        # Platoon coordination topics (same for all vehicles)
        self._platoon_sp_sub = self.create_subscription(
            Float32,
            f"/platoon/{platoon_id}/setpoint",
            self.platoon_sp_cb,
            qos_pubs,
        )
        self._platoon_mode_sub = self.create_subscription(
            Bool,
            f"{self._name}/state/platoon_enabled",
            self.platoon_mode_cb,
            qos_pubs,
        )

        # Publishers (unchanged)
        self._throttle_pub = self.create_publisher(
            Float32,
            f"{self._name}/command/throttle",
            qos_pubs,
        )
        self._brake_pub = self.create_publisher(
            Float32,
            f"{self._name}/command/brake",
            qos_pubs,
        )

        self._timer = self.create_timer(control_period, self._control_loop)

    # --- Callbacks for state topics ---

    def speed_cb(self, msg: Float32):
        self._speed = float(msg.data)

    def setpoint_cb(self, msg: Float32):
        self._setpoint = float(msg.data)

    def dist_to_veh_cb(self, msg: Float32):
        self._dist_to_veh = float(msg.data)

    def platoon_sp_cb(self, msg: Float32):
        self._platoon_sp = float(msg.data)

    def platoon_mode_cb(self, msg: Bool):
        self._platoon_enabled = bool(msg.data)

    # --- Core control logic (CARLA-agnostic) ---

    def compute_speed_setpoint(self) -> float:
        """
        Compute the effective speed setpoint for this member.

        Rules:
          * Base setpoint:
              - platoon_enabled == False -> local ACC setpoint (_setpoint)
              - platoon_enabled == True  -> global platoon setpoint (_platoon_sp)
          * Distance logic always applies if there is a valid front vehicle
            (dist_to_veh > 0), but:
              - ACC / lead: only slow down (never increase above base_sp)
              - platoon follower: symmetric (can speed up or slow down)
        """
        speed = self._speed

        # Choose base setpoint
        if self._platoon_enabled:
            base_sp = self._platoon_sp
        else:
            base_sp = self._setpoint

        # If we have no information about a vehicle ahead, use base setpoint
        if self._dist_to_veh <= 0.0:
            return base_sp

        # Compute spacing error
        desired_dist = speed * self._desired_time_headway + self._min_spacing
        desired_dist = max(self._min_spacing + 3.0, desired_dist)
        dist_err = self._dist_to_veh - desired_dist

        # --- Not in platoon: pure ACC behavior, only slow down when too close ---
        if not self._platoon_enabled:
            if dist_err < 0.0:
                # Too close: reduce setpoint (braking more important)
                d_sp = 2.0 * self._K_dist * dist_err
            else:
                # Far enough or opening gap: do not increase beyond base setpoint
                d_sp = 0.0
            return base_sp + d_sp

        # --- In platoon ---
        if self._role == "lead":
            # Platoon lead: track global setpoint, but still ensure safety by
            # reducing setpoint if too close; never increase above base_sp
            if dist_err < 0.0:
                d_sp = 2.0 * self._K_dist * dist_err
            else:
                d_sp = 0.0
            return base_sp + d_sp

        # Follower in platoon: apply symmetric spacing correction (as before)
        if dist_err >= 0.0:
            d_sp = self._K_dist * dist_err
        else:
            d_sp = 2.0 * self._K_dist * dist_err

        return base_sp + d_sp

    def compute_control(self):
        """
        Run PID on speed to get throttle/brake commands in [0, 1].
        Returns (throttle, brake).
        """
        speed_sp = self.compute_speed_setpoint()
        speed_meas = self._speed

        u = self._pid.step(speed_sp, speed_meas)

        if u > 0.0:
            throttle = float(u)
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(abs(u))

        return throttle, brake

    def _control_loop(self):
        """
        Periodic timer callback: compute and publish throttle/brake commands.
        """
        throttle, brake = self.compute_control()

        self._throttle_pub.publish(Float32(data=throttle))
        self._brake_pub.publish(Float32(data=brake))
