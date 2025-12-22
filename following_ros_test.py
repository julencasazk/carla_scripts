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
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32, Bool

from PID import PID
from PlatooningROS import PlatoonMember


def closest_spawn_point(spawn_points, loc: carla.Location) -> carla.Transform:
    """Return the spawn point (Transform) whose location is closest to loc."""
    best_sp = None
    best_dist2 = float("inf")
    for sp in spawn_points:
        dx = sp.location.x - loc.x
        dy = sp.location.y - loc.y
        dz = sp.location.z - loc.z
        d2 = dx * dx + dy * dy + dz * dz
        if d2 < best_dist2:
            best_dist2 = d2
            best_sp = sp
    return best_sp


def lane_keep_steer(
    vehicle: carla.Vehicle,
    carla_map: carla.Map,
    lookahead_m: float = 12.0,
    steer_gain: float = 1.0,
    wheelbase_m: float = 2.8,
) -> float:
    """
    Simple lane-keeping steering using a pure-pursuit style lookahead waypoint.
    Returns normalized steer in [-1, 1].
    """
    tf = vehicle.get_transform()
    wp = carla_map.get_waypoint(
        tf.location, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if wp is None:
        return 0.0

    next_wps = wp.next(max(1.0, float(lookahead_m)))
    if not next_wps:
        return 0.0

    target = next_wps[0].transform.location
    vec = target - tf.location

    forward = tf.get_forward_vector()
    right = tf.get_right_vector()
    x = vec.x * forward.x + vec.y * forward.y + vec.z * forward.z
    y = vec.x * right.x + vec.y * right.y + vec.z * right.z

    if x <= 1e-3:
        return 0.0

    curvature = (2.0 * y) / (x * x + y * y)
    steer_angle = math.atan(wheelbase_m * curvature)
    return float(np.clip(steer_gain * steer_angle, -1.0, 1.0))


class FollowingRosTest(Node):

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
        self.platoon_mode_pubs = {}

        # QoS profiles:
        # - State streams @100 Hz: "latest sample" semantics, no backlog.
        # - Setpoint/mode: do not drop, and allow late-joiners to receive last value.
        # - Commands: do not drop actuator commands.
        self.qos_state = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.qos_setpoint = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.qos_cmd = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # Global platoon setpoint publisher
        self.platoon_sp_pub = self.create_publisher(
            Float32, "/platoon/plat_0/setpoint", self.qos_setpoint
        )

        # Latest commands from PlatoonMember nodes (name -> dict)
        self._last_cmd = {}

        # Latest local speed setpoints from PlatoonMember nodes (name -> float)
        self._last_local_sp = {}

        # Per-vehicle spacing policy parameters (to mirror PlatoonMember)
        # and allow computing desired distances in this bridge.
        self._desired_time_headways = {}
        self._min_spacings = {}

        # CARLA setup
        self.get_logger().info("Connecting to CARLA...")
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        client.load_world("Town04")
        time.sleep(2.0)
        world = client.get_world()
        self.world = world
        self.carla_map = world.get_map()
        bp_library = world.get_blueprint_library()

        # Save original settings so we can restore on shutdown (matches python-only script).
        self.original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.Ts
        settings.no_rendering_mode = False
        #settings.substepping = True
        #settings.max_substep_delta_time = 0.001
        #settings.max_substeps = 10 # Trying this to hopefully make the physics more reliable
        world.apply_settings(settings)
        self.get_logger().info(f"Synchronous mode ON, dt = {self.Ts} s")

        spawn_points = world.get_map().get_spawn_points()
        # Match following_python_test.py spawn selection and offset.
        base_spawn_point = spawn_points[125]
        base_spawn_point.location.y -= 60.0  # put the leader further ahead to let the followers have space to spawn
        self._spawn_points = spawn_points
        self._base_spawn_point = base_spawn_point

        vehicle_bp = bp_library.find("vehicle.tesla.model3")

        # Distance between vehicles at spawn so they don't collide immediately
        self.initial_spacing = 6.0

        self.vehicles = []

        # Spawn platoon members
        for i in range(self.plen):
            ros_name = "veh_lead" if i == 0 else f"veh_{i}"

            spawn_tf = carla.Transform(
                location=carla.Location(
                    x=base_spawn_point.location.x,
                    y=base_spawn_point.location.y + i * self.initial_spacing,
                    z=base_spawn_point.location.z,
                ),
                rotation=base_spawn_point.rotation,
            )
            veh = world.spawn_actor(vehicle_bp, spawn_tf)
            veh.set_autopilot(False)

            self.get_logger().info(f"Spawned {veh.type_id} as {ros_name} at {spawn_tf}")

            self.ros_names.append(ros_name)
            self.vehicles.append(veh)

            # Store spacing parameters (mirror of PlatoonMember construction)
            # so we can compute desired distances here for logging.
            self._desired_time_headways[ros_name] = 0.7 if i == 1 else 0.3
            self._min_spacings[ros_name] = 7.5

            # State publishers
            self.speed_pubs[ros_name] = self.create_publisher(
                Float32, f"{ros_name}/state/speed", self.qos_state
            )
            self.dist_pubs[ros_name] = self.create_publisher(
                Float32, f"{ros_name}/state/dist_to_veh", self.qos_state
            )
            self.platoon_mode_pubs[ros_name] = self.create_publisher(
                Bool, f"{ros_name}/state/platoon_enabled", self.qos_setpoint
            )
            # Publish once (TRANSIENT_LOCAL) to avoid sending extra messages at 100 Hz.
            self.platoon_mode_pubs[ros_name].publish(Bool(data=True))

            # Command subscribers; controllers are PlatoonMember nodes
            self._last_cmd[ros_name] = {"throttle": 0.0, "brake": 0.0}
            self._last_local_sp[ros_name] = 0.0

            self.create_subscription(
                Float32,
                f"{ros_name}/command/throttle",
                lambda msg, n=ros_name: self._throttle_cb(msg, n),
                self.qos_cmd,
            )
            self.create_subscription(
                Float32,
                f"{ros_name}/command/brake",
                lambda msg, n=ros_name: self._brake_cb(msg, n),
                self.qos_cmd,
            )

            # Local setpoint subscribers: each PlatoonMember publishes its
            # own local speed setpoint on <name>/state/local_setpoint
            self.create_subscription(
                Float32,
                f"{ros_name}/state/local_setpoint",
                lambda msg, n=ros_name: self._local_sp_cb(msg, n),
                self.qos_state,
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

        # CSV logging (all vehicles)
        csv_filename = args.f
        self.csv_file = open(csv_filename, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        # Build header: time_s, then for each vehicle a block of
        # throttle_<name>, brake_<name>, speed_<name>, accel_x_<name>,
        # local_sp_<name>, dist_to_prev_<name>, desired_dist_<name>
        header = ["time_s"]
        for i, name in enumerate(self.ros_names):
            suffix = "lead" if i == 0 else str(i)
            header.extend([
                f"throttle_{suffix}",
                f"brake_{suffix}",
                f"speed_{suffix}",
                f"accel_x_{suffix}",
                f"local_sp_{suffix}",
                f"dist_to_prev_{suffix}",
                f"desired_dist_{suffix}",
            ])
        self.csv_writer.writerow(header)
        self.csv_file.flush()
        self.get_logger().info(f"Logging to: {csv_filename}")

        # Step test definition (inspired by original platooning script)
        self.dt = settings.fixed_delta_seconds
        self.total_time = 3600.0
        self.total_steps = int(self.total_time / self.dt)

        # Distance-based teleport (matches following_python_test.py behavior)
        self.teleport_dist_m = float(args.teleport_dist)
        self.dist_since_tp_m = 0.0
        self.last_lead_loc = None

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
        # Minimum time between setpoint changes (in steps)
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

    def _local_sp_cb(self, msg: Float32, name: str) -> None:
        value = float(msg.data)
        if name in self._last_local_sp:
            self._last_local_sp[name] = value

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

        # Distance-based teleport: when leader has traveled N meters, teleport
        # whole platoon back while preserving geometry (spawn snapping for z).
        if self.teleport_dist_m > 0.0:
            lead_loc = self.vehicles[0].get_transform().location
            if self.last_lead_loc is None:
                self.last_lead_loc = lead_loc
            else:
                self.dist_since_tp_m += math.dist(
                    (lead_loc.x, lead_loc.y, lead_loc.z),
                    (self.last_lead_loc.x, self.last_lead_loc.y, self.last_lead_loc.z),
                )
                self.last_lead_loc = lead_loc

            if self.dist_since_tp_m >= self.teleport_dist_m:
                self.get_logger().info(
                    f"Teleport triggered: leader traveled {self.dist_since_tp_m:.2f} m "
                    f"(threshold {self.teleport_dist_m:.2f} m)"
                )
                lead_tf = self.vehicles[0].get_transform()
                base_tf = self._base_spawn_point
                ref_rotation = base_tf.rotation

                for v in self.vehicles:
                    tf = v.get_transform()
                    rel = tf.location - lead_tf.location

                    target_loc = carla.Location(
                        x=base_tf.location.x + rel.x,
                        y=base_tf.location.y + rel.y,
                        z=base_tf.location.z + rel.z,
                    )
                    sp = closest_spawn_point(self._spawn_points, target_loc)
                    new_loc = carla.Location(
                        x=target_loc.x,
                        y=target_loc.y,
                        z=sp.location.z + 0.1,
                    )
                    v.set_transform(carla.Transform(location=new_loc, rotation=ref_rotation))

                self.dist_since_tp_m = 0.0
                self.last_lead_loc = self.vehicles[0].get_transform().location

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

        # Publish per-vehicle state
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

        # Compute desired distances according to spacing policy used in
        # the Python platooning logic: max(min_spacing + 3.0, v*T + min_spacing)
        desired_dists = []
        for i, name in enumerate(self.ros_names):
            if i == 0:
                # Leader has no vehicle ahead; define 0 for convenience
                desired_dists.append(0.0)
                continue

            th = float(self._desired_time_headways.get(name, 0.3))
            ms = float(self._min_spacings.get(name, 5.0))
            v = float(speeds[i])
            desired_dists.append(max(ms + 3.0, v * th + ms))

        # Step logic to change lead SP when ego settles (matches Python script)
        if self.plen > 1 and (self.step - self.last_change_step) >= self.min_wait_steps:
            ego_speed = float(speeds[self.ego_index])
            if abs(ego_speed - self.last_change_speed) <= self.band:
                if self.lead_sp_idx == (len(self.lead_speed_setpoints) - 1):
                    self.finished = True
                else:
                    self.lead_sp_idx = (self.lead_sp_idx + 1) % len(self.lead_speed_setpoints)
                self.get_logger().info(
                    f"Lead SP change to {self.lead_speed_setpoints[self.lead_sp_idx]:.3f} m/s "
                    f"at t={self.sim_time:.2f}s (ego speed: {ego_speed:.2f} m/s)"
                )
                self.last_change_step = self.step
                self.last_change_speed = ego_speed
            else:
                self.last_change_step = self.step
                self.last_change_speed = ego_speed

        # Logging to CSV: per-vehicle throttle, brake, speed, accel_x,
        # local setpoint, distance to previous, desired distance
        row = [f"{self.sim_time:.4f}"]
        for i, name in enumerate(self.ros_names):
            cmd = self._last_cmd.get(name, {"throttle": 0.0, "brake": 0.0})
            throttle = float(cmd["throttle"])
            brake = float(cmd["brake"])
            speed_i = float(speeds[i])
            accel_x_i = float(accels[i].x)
            local_sp = float(self._last_local_sp.get(name, sp))
            dist_i = float(dists[i])
            desired_dist_i = float(desired_dists[i])

            row.extend([
                f"{throttle:.4f}",
                f"{brake:.4f}",
                f"{speed_i:.4f}",
                f"{accel_x_i:.4f}",
                f"{local_sp:.4f}",
                f"{dist_i:.4f}",
                f"{desired_dist_i:.4f}",
            ])

        self.csv_writer.writerow(row)
        self.csv_file.flush()

        self.get_logger().info(
            f"Step {self.step}/{self.total_steps} t={self.sim_time:.2f}s SP={sp:.2f}"
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

            steer_cmd = 0.0
            if getattr(self.args, "lane_keep", False):
                steer_cmd = lane_keep_steer(
                    v,
                    self.carla_map,
                    lookahead_m=float(getattr(self.args, "lane_lookahead", 12.0)),
                    steer_gain=float(getattr(self.args, "lane_gain", 1.0)),
                )

            control = carla.VehicleControl(
                throttle=throttle_cmd,
                brake=brake_cmd,
                steer=float(steer_cmd),
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
            self.world.apply_settings(self.original_settings)
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
    parser.add_argument(
        "--teleport-dist",
        type=float,
        default=250.0,
        help="Teleport platoon back when leader travels this distance [m]; set <= 0 to disable.",
    )
    parser.add_argument(
        "--lane-keep",
        action="store_true",
        help="Enable simple lane-keeping steering (pure pursuit) while keeping longitudinal control over ROS.",
    )
    parser.add_argument(
        "--lane-lookahead",
        type=float,
        default=12.0,
        help="Lane-keeping lookahead distance in meters.",
    )
    parser.add_argument(
        "--lane-gain",
        type=float,
        default=1.0,
        help="Lane-keeping steering gain.",
    )
    args = parser.parse_args()

    rclpy.init()
    bridge = FollowingRosTest(args)

    # Create Python ROS platoon controller nodes (PlatoonMember) for all vehicles
    Ts = bridge.Ts

    # PID scheduling gains (match following_python_test.py / Platooning.py)
    kb_aw = 1.0
    u_min, u_max = -1.0, 1.0

    nodes = [bridge]
    for i, name in enumerate(bridge.ros_names):
        role = "lead" if i == 0 else "follower"

        slow_pid = PID(0.43127789, 0.43676547, 0.0, 15.0, Ts, u_min, u_max, kb_aw, der_on_meas=True)
        mid_pid = PID(0.11675119, 0.085938,   0.0, 14.90530836, Ts, u_min, u_max, kb_aw, der_on_meas=True)
        fast_pid = PID(0.13408096, 0.07281374, 0.0, 12.16810135, Ts, u_min, u_max, kb_aw, der_on_meas=True)

        # Use the same spacing parameters as stored in the bridge so that the
        # desired distance used by controllers matches what is logged.
        desired_time_headway = float(bridge._desired_time_headways.get(name, 0.3))
        min_spacing = float(bridge._min_spacings.get(name, 5.0))

        member = PlatoonMember(
            name=name,
            role=role,
            slow_pid=slow_pid,
            mid_pid=mid_pid,
            fast_pid=fast_pid,
            desired_time_headway=desired_time_headway,
            min_spacing=min_spacing,
            K_dist=2.5,
            control_period=Ts,
        )

        # We drive the control loop explicitly via step_once() once per
        # CARLA tick in the main loop below. Disable the internal timer so
        # the PID is stepped exactly at Ts = 0.01 s, matching its tuning.
        try:
            member._timer.cancel()
        except Exception:
            pass

        nodes.append(member)

    executor = MultiThreadedExecutor()
    for n in nodes:
        executor.add_node(n)

    def drain_executor(max_callbacks: int, max_wall_s: float) -> None:
        """Process queued ROS callbacks to minimize state/command staleness."""
        start = time.perf_counter()
        for _ in range(int(max_callbacks)):
            executor.spin_once(timeout_sec=0.0)
            if (time.perf_counter() - start) >= max_wall_s:
                break

    # Ensure CARLA does not tick faster than real-time dt (0.01s):
    # one simulation tick (fixed_delta_seconds) should take >= dt wall time.
    dt_wall_s = float(bridge.dt)
    last_tick_wall_s = None
    drain_budget_s = min(0.003, dt_wall_s * 0.30)

    # Main simulation loop at 100 Hz sim time:
    #  1) Apply controls from previous tick
    #  2) Tick CARLA and publish state/setpoints/mode
    #  3) Spin executor to deliver state to controllers
    #  4) Run each PlatoonMember control step once (publish commands)
    #  5) Spin executor to deliver commands back to bridge
    try:
        while not bridge.finished:
            if last_tick_wall_s is not None:
                next_tick_wall_s = last_tick_wall_s + dt_wall_s
                now_s = time.perf_counter()
                if now_s < next_tick_wall_s:
                    time.sleep(next_tick_wall_s - now_s)

            # 1) Apply controls just before advancing CARLA
            bridge.apply_controls()

            # 2) Advance CARLA one synchronous step and publish state
            bridge.step_simulation()
            last_tick_wall_s = time.perf_counter()

            # 3) Deliver state and setpoint messages to PlatoonMember nodes
            drain_executor(max_callbacks=500, max_wall_s=drain_budget_s)

            # 4) Run one control update per PlatoonMember (deterministic per-tick)
            for n in nodes:
                if isinstance(n, PlatoonMember):
                    n.step_once()

            # 5) Deliver throttle/brake commands back to bridge subscribers
            drain_executor(max_callbacks=500, max_wall_s=drain_budget_s)
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
