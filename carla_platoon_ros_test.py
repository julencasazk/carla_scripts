import argparse
import math
import time

import carla
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32

from PID import PID
from PlatooningROS import PlatoonMember


class CarlaPlatoonBridge(Node):
    """Single-process CARLA–ROS platoon bridge.

    - Spawns a platoon of CARLA vehicles.
    - Publishes per-vehicle state topics:
        <name>/state/speed       (Float32)
        <name>/state/dist_to_veh (Float32)
    - Subscribes to per-vehicle command topics:
        <name>/command/throttle  (Float32)
        <name>/command/brake     (Float32)
    - Applies commands to CARLA vehicles each simulation tick.

    All controller logic (PID, platoon policy) is in PlatoonMember or the MCU.
    """

    def __init__(
        self,
        host: str,
        port: int,
        plen: int,
        mcu_index: int,
        dt: float,
    ) -> None:
        super().__init__("carla_platoon_bridge")

        if plen < 1:
            raise ValueError("Platoon length must be >= 1")
        if not (0 <= mcu_index < plen):
            raise ValueError("mcu_index must be in [0, plen-1]")

        self._plen = plen
        self._mcu_index = mcu_index
        self._dt = dt

        # --- Connect to CARLA and configure world ---
        self.get_logger().info("Connecting to CARLA...")
        client = carla.Client(host, port)
        client.set_timeout(10.0)

        client.load_world("Town04")
        time.sleep(2.0)
        world = client.get_world()
        self._world = world

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = dt
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        self.get_logger().info(f"Synchronous mode ON, dt = {dt} s")

        bp_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        base_spawn_point = spawn_points[54]
        base_spawn_point.location.z -= 0.3

        vehicle_bp = bp_library.find("vehicle.tesla.model3")
        initial_spacing = 5.0

        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE)

        self._veh_names = []          # logical names: veh_lead, veh_ego, veh_1, ...
        self._vehicles = []           # CARLA actor handles
        self._speed_pubs = {}         # name -> Publisher(Float32)
        self._dist_pubs = {}          # name -> Publisher(Float32)
        self._last_cmd = {}           # name -> {"throttle": float, "brake": float}

        # --- Spawn vehicles and set up ROS topics ---
        for i in range(plen):
            # Naming scheme:
            #  - mcu_index is always "veh_ego" (even if it's also the lead)
            #  - index 0 (if not MCU) is "veh_lead"
            #  - others are "veh_<i>"
            if i == mcu_index:
                ros_name = "veh_ego"
            elif i == 0:
                ros_name = "veh_lead"
            else:
                ros_name = f"veh_{i}"

            spawn_tf = carla.Transform(
                location=carla.Location(
                    x=base_spawn_point.location.x + i * initial_spacing,
                    y=base_spawn_point.location.y,
                    z=base_spawn_point.location.z,
                ),
                rotation=base_spawn_point.rotation,
            )
            veh = world.spawn_actor(vehicle_bp, spawn_tf)
            veh.set_autopilot(False)

            self.get_logger().info(f"Spawned {veh.type_id} as {ros_name} at {spawn_tf}")

            self._veh_names.append(ros_name)
            self._vehicles.append(veh)

            self._speed_pubs[ros_name] = self.create_publisher(
                Float32, f"{ros_name}/state/speed", qos
            )
            self._dist_pubs[ros_name] = self.create_publisher(
                Float32, f"{ros_name}/state/dist_to_veh", qos
            )

            self._last_cmd[ros_name] = {"throttle": 0.0, "brake": 0.0}

            # Command subscribers; controller may be PlatoonMember or MCU
            self.create_subscription(
                Float32,
                f"{ros_name}/command/throttle",
                lambda msg, n=ros_name: self._throttle_cb(msg, n),
                qos,
            )
            self.create_subscription(
                Float32,
                f"{ros_name}/command/brake",
                lambda msg, n=ros_name: self._brake_cb(msg, n),
                qos,
            )

        # Timer driving the synchronous CARLA stepping and bridge I/O
        self._timer = self.create_timer(self._dt, self._step)

    # --- Command callbacks ---

    def _throttle_cb(self, msg: Float32, name: str) -> None:
        value = float(msg.data)
        self._last_cmd[name]["throttle"] = value
        # Debug: confirm commands are received
        self.get_logger().info(f"[CB throttle] {name} <- {value:.2f}")

    def _brake_cb(self, msg: Float32, name: str) -> None:
        value = float(msg.data)
        self._last_cmd[name]["brake"] = value
        self.get_logger().info(f"[CB brake] {name} <- {value:.2f}")

    # --- Main bridge step ---

    def _step(self) -> None:
        # Advance CARLA by one synchronous tick
        self._world.tick()
        snapshot = self._world.get_snapshot()
        t = snapshot.timestamp.elapsed_seconds

        speeds = []
        transforms = []
        for veh in self._vehicles:
            vel = veh.get_velocity()
            tf = veh.get_transform()
            speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            speeds.append(speed)
            transforms.append(tf)

        dists = []
        for i, name in enumerate(self._veh_names):
            speed = speeds[i]
            self._speed_pubs[name].publish(Float32(data=float(speed)))

            if i == 0:
                dist_to_prev = 0.0
            else:
                prev_tf = transforms[i - 1]
                tf = transforms[i]
                dist_to_prev = math.dist(
                    (tf.location.x, tf.location.y, tf.location.z),
                    (prev_tf.location.x, prev_tf.location.y, prev_tf.location.z),
                )
            self._dist_pubs[name].publish(Float32(data=float(dist_to_prev)))
            dists.append(dist_to_prev)

        # Apply last commands to CARLA and log
        for idx, (name, veh) in enumerate(zip(self._veh_names, self._vehicles)):
            cmd = self._last_cmd[name]
            throttle = max(0.0, min(1.0, cmd["throttle"]))
            brake = max(0.0, min(1.0, cmd["brake"]))

            ctrl = carla.VehicleControl(
                throttle=float(throttle),
                brake=float(brake),
                steer=0.0,
                hand_brake=False,
                reverse=False,
            )
            veh.apply_control(ctrl)

            self.get_logger().info(
                f"[CARLA {name}] t={t:.2f}s v={speeds[idx]:.2f} m/s "
                f"dist_prev={dists[idx]:.2f} m throttle={throttle:.2f} brake={brake:.2f}"
            )

    def destroy(self) -> None:
        self.get_logger().info("Destroying CARLA actors...")
        for v in self._vehicles:
            try:
                v.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            except Exception:
                pass
        time.sleep(1.0)
        for v in self._vehicles:
            try:
                v.destroy()
            except Exception:
                pass
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description="CARLA–ROS platoon bridge (single process)")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--plen", type=int, default=3, help="Number of platoon members")
    parser.add_argument(
        "--mcu-index",
        type=int,
        default=1,
        help="Index [0..plen-1] of the MCU (micro-ROS) vehicle in the platoon",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Control / CARLA step [s]")
    args = parser.parse_args()

    rclpy.init()

    # Bridge node (CARLA + ROS topics)
    bridge = CarlaPlatoonBridge(
        host=args.host,
        port=args.port,
        plen=args.plen,
        mcu_index=args.mcu_index,
        dt=args.dt,
    )

    nodes = [bridge]

    # PID gains (you can tune these as before in following_python_test)
    kp = 0.26
    ki = 0.13
    kd = 0.006
    N_d = 5.0
    u_min, u_max = -1.0, 1.0
    kb_aw = 1.0

    # Create ROS PlatoonMember controllers for all NON-MCU vehicles
    for i, name in enumerate(bridge._veh_names):
        if i == args.mcu_index:
            # Controlled externally by STM32 via /veh_ego/command/*
            continue

        role = "lead" if i == 0 else "follower"

        pid = PID(
            kp,
            ki,
            kd,
            N_d,
            args.dt,
            u_min,
            u_max,
            kb_aw,
            der_on_meas=True,
        )

        node = PlatoonMember(
            name=name,
            role=role,
            pid=pid,
            desired_time_headway=0.7 if i == 1 else 0.3,
            min_spacing=5.0,
            K_dist=0.3,
            control_period=args.dt,
        )
        nodes.append(node)

    executor = MultiThreadedExecutor()
    for n in nodes:
        executor.add_node(n)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for n in nodes:
            if isinstance(n, CarlaPlatoonBridge):
                n.destroy()
            else:
                n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
