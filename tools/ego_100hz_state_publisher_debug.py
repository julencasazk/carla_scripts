"""
Publishes ego speed/dist/setpoint at high rate from CARLA for timing debug.

Outputs
  - ROS topics at configured rate.

Run (example)
  python3 tools/ego_100hz_state_publisher_debug.py --rate-hz 100
"""

import argparse
import math
import time
from typing import Optional, Tuple

import carla
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32


def speed_mps(vehicle: carla.Vehicle) -> float:
    vel = vehicle.get_velocity()
    return float(math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z))


def distance_m(a: carla.Vehicle, b: carla.Vehicle) -> float:
    ta = a.get_transform()
    tb = b.get_transform()
    return float(
        math.dist(
            (ta.location.x, ta.location.y, ta.location.z),
            (tb.location.x, tb.location.y, tb.location.z),
        )
    )


def _set_role_name(bp: carla.ActorBlueprint, role_name: str) -> None:
    try:
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", role_name)
    except Exception:
        pass


class EgoStatePublisher(Node):
    def __init__(
        self,
        ego_vehicle: carla.Vehicle,
        lead_vehicle: Optional[carla.Vehicle],
        setpoint_mps: float,
        rate_hz: float,
        log_every: int,
    ) -> None:
        super().__init__("ego_100hz_state_publisher_debug")
        self.ego_vehicle = ego_vehicle
        self.lead_vehicle = lead_vehicle
        self.setpoint_mps = float(setpoint_mps)
        self.period_s = 1.0 / float(rate_hz)
        self.log_every = int(log_every)

        qos_state = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.pub_speed = self.create_publisher(Float32, "veh_ego/state/speed", qos_state)
        self.pub_dist = self.create_publisher(Float32, "veh_ego/state/dist_to_veh", qos_state)
        self.pub_setpoint = self.create_publisher(Float32, "veh_ego/state/setpoint", qos_state)

        self._last_tick_wall_s: Optional[float] = None
        self._count = 0

        self.create_timer(self.period_s, self._on_timer)

    def _on_timer(self) -> None:
        now_s = time.perf_counter()
        if self._last_tick_wall_s is not None and self.log_every > 0:
            if (self._count % self.log_every) == 0:
                dt_s = now_s - self._last_tick_wall_s
                hz = (1.0 / dt_s) if dt_s > 0.0 else float("inf")
                self.get_logger().info(
                    f"timer_dt={dt_s:.6f}s (~{hz:.1f} Hz), target={self.period_s:.6f}s"
                )
        self._last_tick_wall_s = now_s
        self._count += 1

        try:
            v = speed_mps(self.ego_vehicle)
        except Exception:
            v = 0.0

        if self.lead_vehicle is not None:
            try:
                dist = distance_m(self.ego_vehicle, self.lead_vehicle)
            except Exception:
                dist = 0.0
        else:
            dist = 0.0

        self.pub_speed.publish(Float32(data=float(v)))
        self.pub_dist.publish(Float32(data=float(dist)))
        self.pub_setpoint.publish(Float32(data=float(self.setpoint_mps)))


def spawn_vehicles(
    world: carla.World,
    plen: int,
    base_spawn_index: int,
    spacing_m: float,
) -> Tuple[carla.Vehicle, Optional[carla.Vehicle], list]:
    bp_library = world.get_blueprint_library()
    vehicle_bp = bp_library.find("vehicle.tesla.model3")

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found in current CARLA map.")
    if not (0 <= base_spawn_index < len(spawn_points)):
        raise ValueError(f"--spawn-index must be in [0, {len(spawn_points) - 1}]")

    base_tf = spawn_points[base_spawn_index]
    base_tf.location.y -= 60.0

    vehicles = []
    for i in range(max(1, int(plen))):
        role = "veh_lead" if i == 0 else ("veh_ego" if i == 1 else f"veh_{i}")
        _set_role_name(vehicle_bp, role)
        spawn_tf = carla.Transform(
            location=carla.Location(
                x=base_tf.location.x,
                y=base_tf.location.y + i * float(spacing_m),
                z=base_tf.location.z,
            ),
            rotation=base_tf.rotation,
        )
        v = world.try_spawn_actor(vehicle_bp, spawn_tf)
        if v is None:
            raise RuntimeError(f"Failed to spawn {role} at {spawn_tf}")
        v.set_autopilot(False)
        vehicles.append(v)

    lead = vehicles[0]
    ego = vehicles[1] if len(vehicles) > 1 else vehicles[0]
    return ego, (lead if lead.id != ego.id else None), vehicles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug: publish veh_ego speed/dist/setpoint at 100 Hz (ROS2 only)."
    )
    parser.add_argument("--host", type=str, default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--rate-hz", type=float, default=100.0, help="Publish rate (Hz)")
    parser.add_argument("--setpoint", type=float, default=0.0, help="Constant setpoint (m/s)")
    parser.add_argument("--log-every", type=int, default=100, help="Log timer dt every N callbacks")
    parser.add_argument("--spawn", action="store_true", help="Spawn lead+ego vehicles in CARLA")
    parser.add_argument("--plen", type=int, default=2, help="If --spawn: number of vehicles to spawn (>=1)")
    parser.add_argument("--spawn-index", type=int, default=125, help="If --spawn: CARLA spawn point index")
    parser.add_argument("--spacing", type=float, default=6.0, help="If --spawn: spacing between spawned vehicles (m)")
    args = parser.parse_args()

    client = carla.Client(args.host, int(args.port))
    client.set_timeout(10.0)
    world = client.get_world()

    original_settings = world.get_settings()
    vehicles = []

    if args.spawn:
        ego, lead, vehicles = spawn_vehicles(
            world=world,
            plen=int(args.plen),
            base_spawn_index=int(args.spawn_index),
            spacing_m=float(args.spacing),
        )
    else:
        # Best-effort: pick any existing vehicle as ego.
        actors = world.get_actors().filter("vehicle.*")
        if not actors:
            raise RuntimeError("No vehicles in CARLA world. Use --spawn or spawn vehicles elsewhere.")
        ego = actors[0]
        lead = None

    rclpy.init()
    node = EgoStatePublisher(
        ego_vehicle=ego,
        lead_vehicle=lead,
        setpoint_mps=float(args.setpoint),
        rate_hz=float(args.rate_hz),
        log_every=int(args.log_every),
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
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
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass
        for v in vehicles:
            try:
                v.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    main()
