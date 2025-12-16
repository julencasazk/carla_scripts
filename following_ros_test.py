import argparse
import math
import time
import csv
import threading
import queue

import carla
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32, Bool

from PID import PID
from PlatooningROS import PlatoonMember


class FollowingRosTest(Node):
    """ROS2 platoon longitudinal test (multi-vehicle version).

    - Connects to CARLA in synchronous mode (dt = Ts).
    - Spawns a platoon of Tesla Model 3s of length --plen at spawn point 54.
    - Publishes per-vehicle state topics for CARLA-agnostic PlatoonMember nodes:
            <name>/state/speed
            <name>/state/dist_to_veh
            <name>/state/setpoint
            <name>/state/platoon_enabled
        and a global platoon setpoint:
            /platoon/plat_0/setpoint
    - Subscribes to per-vehicle command topics from PlatoonMember nodes:
            <name>/command/throttle
            <name>/command/brake
        and applies them to CARLA vehicles.
    - Runs a step-test sequence of lead speed setpoints (like the original
        platooning script), advancing the global setpoint when the ego member's
        speed settles within a band.
    - Attaches a dashcam to the ego vehicle and displays it via OpenCV in a
        background thread.
    - Logs ego data to CSV (time, throttle, setpoint, brake, speed, accel).
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("following_ros_test")

        self.args = args

        if args.plen < 1:
            raise ValueError("Platoon length (--plen) must be >= 1")

        # Sampling time / CARLA dt
        self.Ts = 0.01

        # Platoon configuration
        self.plen = int(args.plen)
        # index of ego vehicle for logging/dashcam: last in platoon if >1, else leader
        self.ego_index = self.plen - 1 if self.plen > 1 else 0

        # Per-vehicle ROS publishers (filled after spawning vehicles)
        self.ros_names = []
        self.speed_pubs = {}
        self.dist_pubs = {}
        self.setpoint_pubs = {}
        self.platoon_mode_pubs = {}

        # Global platoon setpoint publisher
        self.platoon_sp_pub = self.create_publisher(Float32, "/platoon/plat_0/setpoint", 10)

        # Latest commands from PlatoonMember nodes (name -> dict)
        self._last_cmd = {}

        # CARLA setup
        self.get_logger().info("Connecting to CARLA...")
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        client.load_world("Town04")
        time.sleep(2.0)
        world = client.get_world()
        self.world = world
        bp_library = world.get_blueprint_library()

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.Ts
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        self.get_logger().info(f"Synchronous mode ON, dt = {self.Ts} s")

        spawn_points = world.get_map().get_spawn_points()
        base_spawn_point = spawn_points[54]
        base_spawn_point.location.z -= 0.3

        vehicle_bp = bp_library.find("vehicle.tesla.model3")

        # Distance between vehicles at spawn so they don't collide immediately
        self.initial_spacing = 10.0

        self.vehicles = []

        # Spawn platoon members
        for i in range(self.plen):
            ros_name = "veh_lead" if i == 0 else f"veh_{i}"

            spawn_tf = carla.Transform(
                location=carla.Location(
                    x=base_spawn_point.location.x + i * self.initial_spacing,
                    y=base_spawn_point.location.y,
                    z=base_spawn_point.location.z,
                ),
                rotation=base_spawn_point.rotation,
            )
            veh = world.spawn_actor(vehicle_bp, spawn_tf)
            veh.set_autopilot(False)

            self.get_logger().info(f"Spawned {veh.type_id} as {ros_name} at {spawn_tf}")

            self.ros_names.append(ros_name)
            self.vehicles.append(veh)

            # State publishers
            self.speed_pubs[ros_name] = self.create_publisher(
                Float32, f"{ros_name}/state/speed", 10
            )
            self.dist_pubs[ros_name] = self.create_publisher(
                Float32, f"{ros_name}/state/dist_to_veh", 10
            )
            self.setpoint_pubs[ros_name] = self.create_publisher(
                Float32, f"{ros_name}/state/setpoint", 10
            )
            self.platoon_mode_pubs[ros_name] = self.create_publisher(
                Bool, f"{ros_name}/state/platoon_enabled", 10
            )

            # Command subscribers; controllers are PlatoonMember nodes
            self._last_cmd[ros_name] = {"throttle": 0.0, "brake": 0.0}

            self.create_subscription(
                Float32,
                f"{ros_name}/command/throttle",
                lambda msg, n=ros_name: self._throttle_cb(msg, n),
                10,
            )
            self.create_subscription(
                Float32,
                f"{ros_name}/command/brake",
                lambda msg, n=ros_name: self._brake_cb(msg, n),
                10,
            )

        # Dashcam infrastructure: queue + worker thread (non-blocking viewer)
        # Avoid Python 3.10+ type union syntax for compatibility
        self._img_queue = None  # type: ignore[assignment]
        self._dashcam_thread = None  # type: ignore[assignment]
        self._dashcam_stop = threading.Event()

        # Dashcam on ego vehicle (like original script, but in-thread viewer)
        self.camera = None
        try:
            self._img_queue = queue.Queue(maxsize=1)

            cam_bp = bp_library.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", "640")
            cam_bp.set_attribute("image_size_y", "480")
            cam_bp.set_attribute("fov", "90")

            cam_tf = carla.Transform(carla.Location(x=0.0, y=10.0, z=10.0), carla.Rotation(yaw=-20.0, pitch=-35))
            ego_vehicle = self.vehicles[self.ego_index]
            self.camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego_vehicle)
            self.get_logger().info(
                f"Spawned {self.camera.type_id} attached to {self.ros_names[self.ego_index]} (dashcam)"
            )
            self.camera.listen(self._camera_callback)

            # Start dashcam viewer thread
            self._dashcam_thread = threading.Thread(
                target=self._dashcam_worker, name="dashcam_viewer", daemon=True
            )
            self._dashcam_thread.start()
        except Exception as e:
            self.get_logger().warn(f"Failed to create dashcam: {e}")

        # CSV logging (ego vehicle only)
        csv_filename = args.f
        self.csv_file = open(csv_filename, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "time_s",
            "throttle",
            "setpoint",
            "brake",
            "speed_m_s",
            "accel_x_m_s2",
            "accel_y_m_s2",
            "accel_z_m_s2",
        ])
        self.csv_file.flush()
        self.get_logger().info(f"Logging to: {csv_filename}")

        # Step test definition (inspired by original platooning script)
        self.dt = settings.fixed_delta_seconds
        self.total_time = 3600.0
        self.total_steps = int(self.total_time / self.dt)

        self.teleport_time = int(5.0 / self.dt)
        # Lead speed setpoints (global platoon reference, m/s)
        self.lead_speed_setpoints = [
            0.0,
            33.0,
            0.0,
            20.0,
            25.0,
            15.0,
            10.0,
            5.0,
            15.0,
            0.0,
        ]

        self.lead_sp_idx = 0
        self.finished = False

        # State for stability-based setpoint changes
        self.sim_time = 0.0
        self.start_timestamp = None
        self.step = 0
        self.last_change_step = 0
        self.last_change_speed = 0.0
        self.band = 0.2
        self.min_wait_steps = int(5.0 / self.dt)

        # Vehicle reset
        for v in self.vehicles:
            v.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        self.get_logger().info(
            f"Running platoon step test (plen={self.plen}): dt={self.dt}s, total_time={self.total_time}s, total_steps={self.total_steps}"
        )

    # --- Command callbacks from PlatoonMember nodes ---

    def _throttle_cb(self, msg: Float32, name: str) -> None:
        value = float(msg.data)
        if name in self._last_cmd:
            self._last_cmd[name]["throttle"] = value

    def _brake_cb(self, msg: Float32, name: str) -> None:
        value = float(msg.data)
        if name in self._last_cmd:
            self._last_cmd[name]["brake"] = value

    def _camera_callback(self, image: carla.Image) -> None:
        """Camera callback: push latest frame into queue (non-blocking)."""
        if self._img_queue is None:
            return

        payload = (bytes(image.raw_data), image.width, image.height)

        # Drop any stale frame and keep only the latest
        try:
            while True:
                self._img_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._img_queue.put_nowait(payload)
        except queue.Full:
            pass

    def _dashcam_worker(self) -> None:
        """OpenCV viewer loop running in a separate thread."""
        time.sleep(1.0)  # small delay to let camera warm up
        try:
            cv2.namedWindow("Following ROS dashcam", cv2.WINDOW_NORMAL)
        except Exception as e:
            self.get_logger().warn(f"Failed to create OpenCV window: {e}")
            return

        try:
            while not self._dashcam_stop.is_set():
                try:
                    payload = self._img_queue.get(timeout=0.05) if self._img_queue else None
                except queue.Empty:
                    continue

                if payload is None:
                    continue

                raw_bytes, width, height = payload
                try:
                    array = np.frombuffer(raw_bytes, dtype=np.uint8)
                    array = np.reshape(array, (height, width, 4))
                    array = array[:, :, :3]
                    cv2.imshow("Following ROS dashcam", array)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except Exception as e:
                    self.get_logger().warn(f"Dashcam frame error: {e}")
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def step_simulation(self) -> None:
        """One synchronous CARLA step: tick world, publish state, update SP, log."""
        # Advance CARLA using controls that were applied just before this tick
        self.world.tick()
        snapshot = self.world.get_snapshot()
        t = snapshot.timestamp.elapsed_seconds

        if self.start_timestamp is None:
            self.start_timestamp = t
        self.sim_time = t - self.start_timestamp

        if self.sim_time >= self.total_time:
            self.finished = True

        # Teleport periodically to keep the platoon on a straight segment
        if self.step > 0 and self.step % self.teleport_time == 0:
            spawn_points = self.world.get_map().get_spawn_points()
            base_spawn_point = spawn_points[54]
            base_spawn_point.location.z -= 0.3

            # Preserve current longitudinal spacing within the platoon.
            # Compute each vehicle's offset along the leader's heading, then
            # re-spawn them at the base spawn point with the same offsets.
            current_transforms = [v.get_transform() for v in self.vehicles]
            leader_loc = current_transforms[0].location

            yaw_rad = math.radians(base_spawn_point.rotation.yaw)
            forward_x = math.cos(yaw_rad)
            forward_y = math.sin(yaw_rad)

            for i, v in enumerate(self.vehicles):
                tf = current_transforms[i]
                rel_x = tf.location.x - leader_loc.x
                rel_y = tf.location.y - leader_loc.y

                # Signed distance along road direction (positive ahead of leader,
                # negative behind), preserving inter-vehicle spacing.
                dist_along = rel_x * forward_x + rel_y * forward_y

                new_loc = carla.Location(
                    x=base_spawn_point.location.x + dist_along * forward_x,
                    y=base_spawn_point.location.y + dist_along * forward_y,
                    z=base_spawn_point.location.z,
                )
                spawn_tf = carla.Transform(location=new_loc, rotation=base_spawn_point.rotation)
                v.set_transform(spawn_tf)

            self.get_logger().info("Teleport platoon!!")

        # Gather vehicle states
        speeds = []
        accels = []
        transforms = []
        for v in self.vehicles:
            vel = v.get_velocity()
            accel = v.get_acceleration()
            tf = v.get_transform()
            speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            speeds.append(speed)
            accels.append(accel)
            transforms.append(tf)

        # Current global lead speed setpoint
        sp = self.lead_speed_setpoints[self.lead_sp_idx]
        self.platoon_sp_pub.publish(Float32(data=float(sp)))

        # Publish per-vehicle state and local setpoints / mode
        dists = []
        for i, name in enumerate(self.ros_names):
            speed = speeds[i]
            self.speed_pubs[name].publish(Float32(data=float(speed)))

            if i == 0:
                dist_to_prev = 0.0
            else:
                prev_tf = transforms[i - 1]
                tf = transforms[i]
                dist_to_prev = math.dist(
                    (tf.location.x, tf.location.y, tf.location.z),
                    (prev_tf.location.x, prev_tf.location.y, prev_tf.location.z),
                )
            self.dist_pubs[name].publish(Float32(data=float(dist_to_prev)))
            dists.append(dist_to_prev)

            # For this test, local ACC setpoint is just the global lead setpoint
            self.setpoint_pubs[name].publish(Float32(data=float(sp)))
            # Always in platoon mode for this test
            self.platoon_mode_pubs[name].publish(Bool(data=True))

        # Ego vehicle index and state
        ego_idx = self.ego_index
        ego_speed = speeds[ego_idx]
        ego_accel = accels[ego_idx]

        # Stability-based setpoint change logic (based on ego speed)
        if (self.step - self.last_change_step) >= self.min_wait_steps:
            if abs(speeds[0] - self.last_change_speed) <= self.band:
                if self.lead_sp_idx == (len(self.lead_speed_setpoints) - 1):
                    self.finished = True
                else:
                    self.lead_sp_idx = (self.lead_sp_idx + 1) % len(self.lead_speed_setpoints)
                self.get_logger().info(
                    f"Lead SP change to {self.lead_speed_setpoints[self.lead_sp_idx]:.3f} m/s "
                    f"at t={self.sim_time:.2f}s (ego speed: {speeds[0]:.2f} m/s)"
                )
            self.last_change_step = self.step
            self.last_change_speed = speeds[0]

        # Ego logging to CSV (commands currently in effect for this step)
        ego_name = self.ros_names[ego_idx]
        ego_cmd = self._last_cmd.get(ego_name, {"throttle": 0.0, "brake": 0.0})
        ego_throttle = float(ego_cmd["throttle"])
        ego_brake = float(ego_cmd["brake"])

        self.csv_writer.writerow([
            f"{self.sim_time:.4f}",
            f"{ego_throttle:.4f}",
            f"{sp:.4f}",
            f"{ego_brake:.4f}",
            f"{ego_speed:.4f}",
            f"{ego_accel.x:.4f}",
            f"{ego_accel.y:.4f}",
            f"{ego_accel.z:.4f}",
        ])
        self.csv_file.flush()

        self.get_logger().info(
            f"Step {self.step}/{self.total_steps} t={self.sim_time:.2f}s "
            f"SP={sp:.2f} ego_v={ego_speed:.2f} Th={ego_throttle:.2f} Br={ego_brake:.2f}"
        )

        self.step += 1

    def apply_controls(self) -> None:
        """Apply latest commands from controllers to CARLA vehicles.

        This should be called exactly once before each world.tick(), so that
        the commands computed on the previous tick are used for the next
        simulation step.
        """
        for name, v in zip(self.ros_names, self.vehicles):
            cmd = self._last_cmd.get(name, {"throttle": 0.0, "brake": 0.0})
            throttle_cmd = max(0.0, min(1.0, float(cmd["throttle"])))
            brake_cmd = max(0.0, min(1.0, float(cmd["brake"])))

            control = carla.VehicleControl(
                throttle=throttle_cmd,
                brake=brake_cmd,
                steer=0.0,
                hand_brake=False,
                reverse=False,
            )
            v.apply_control(control)

    def shutdown(self) -> None:
        self.get_logger().info("Stopping vehicle and cleaning up...")
        # Stop dashcam thread and camera
        try:
            self._dashcam_stop.set()
        except Exception:
            pass
        if getattr(self, "_img_queue", None) is not None:
            try:
                self._img_queue.put_nowait(None)
            except Exception:
                pass
        if getattr(self, "_dashcam_thread", None) is not None:
            try:
                self._dashcam_thread.join(timeout=1.0)
            except Exception:
                pass
        if getattr(self, "camera", None) is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
            try:
                self.camera.destroy()
            except Exception:
                pass
        try:
            for v in self.vehicles:
                try:
                    v.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                except Exception:
                    pass
            time.sleep(1.0)
            for v in self.vehicles:
                try:
                    v.destroy()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self.csv_file.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="CARLA longitudinal identification test (ROS2)")
    parser.add_argument("--host", type=str, help="CARLA host", default="localhost")
    parser.add_argument("--port", type=int, help="CARLA port", default=2000)
    parser.add_argument("-f", type=str, help="Output csv filename", default="out_ros.csv")
    parser.add_argument("--plen", type=int, default=2, help="Number of platoon members (>=1)")
    args = parser.parse_args()

    rclpy.init()
    bridge = FollowingRosTest(args)

    # Create Python ROS platoon controller nodes (PlatoonMember) for all vehicles
    Ts = bridge.Ts

    # PID gains (from original script)
    kp = 2.59397071e-01
    ki = 1.27381733e-01
    kd = 6.21160744e-03
    N_d = 5.0
    kb_aw = 1.0
    u_min, u_max = -1.0, 1.0

    nodes = [bridge]
    for i, name in enumerate(bridge.ros_names):
        role = "lead" if i == 0 else "follower"
        k = 0.3 + (0.1 * i) 

        pid = PID(kp, ki, kd, N_d, Ts, u_min, u_max, kb_aw, der_on_meas=True)

        member = PlatoonMember(
            name=name,
            role=role,
            pid=pid,
            desired_time_headway=0.7 if i == 1 else 0.3,
            min_spacing=5.0,
            K_dist=k,
            control_period=Ts,
        )
        nodes.append(member)

    executor = MultiThreadedExecutor()
    for n in nodes:
        executor.add_node(n)

    # Main simulation loop at 100 Hz sim time:
    #  1) Apply controls from previous tick
    #  2) Tick CARLA and publish state/setpoints/mode
    #  3) Spin executor to deliver state to controllers
    #  4) Run each PlatoonMember control step once (publish commands)
    #  5) Spin executor to deliver commands back to bridge
    try:
        while not bridge.finished:
            # 1) Apply controls just before advancing CARLA
            bridge.apply_controls()

            # 2) Advance CARLA one synchronous step and publish state
            bridge.step_simulation()

            # 3) Deliver state and setpoint messages to PlatoonMember nodes
            executor.spin_once(timeout_sec=0.0)

            # 4) Run one control update per PlatoonMember (deterministic per-tick)
            for n in nodes:
                if isinstance(n, PlatoonMember):
                    n.step_once()

            # 5) Deliver throttle/brake commands back to bridge subscribers
            executor.spin_once(timeout_sec=0.0)
    except KeyboardInterrupt:
        bridge.get_logger().info("KeyboardInterrupt: stopping test early.")
    finally:
        bridge.shutdown()
        for n in nodes:
            if n is not bridge:
                n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
