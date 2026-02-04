"""
FINAL / CURRENT MAIN ENTRYPOINT
This is the most recent and working platoon experiment script in this project.

What it does
  - Spawns a multi-vehicle platoon in CARLA and runs a standardized speed plan.
  - Uses the CAM-based platoon controller (cooperative braking + speed sharing).
  - Publishes VehicleState topics and a global platoon setpoint.
  - Supports one MCU/HIL vehicle (optional) and can gate start on first throttle.
  - Uses lane-keeping steering and injects a deterministic emergency-brake event.

Outputs
  - CSV logs with per-vehicle state, commands, and timing diagnostics.
  - Console progress and summary messages.

Run (example)
  python3 core/following_ros_cam.py --host localhost --port 2000 --plen 3

"""

import argparse
import math
import random
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
from platooning_msgs.msg import VehicleState

from lib.PID import PID
from lib.PlatooningROSCAM import PlatoonMember

#
# Test orchestration defaults (edit here for thesis-style standardized tests)
#
STABLE_BAND_MPS = 0.5
STABLE_HOLD_S = 5.0
MIN_SEGMENT_S = 2.0
MAX_SEGMENT_S = 90.0

EMERGENCY_TARGET_SP_MPS = 22.0
EMERGENCY_BRAKE_CMD = 1.0
EMERGENCY_BRAKE_DURATION_S = 3.0
EMERGENCY_RECOVERY_S = 10.0

DEFAULT_SETPOINT_PLAN = [
    ("start_stop", 0.0),
    # Low range (0-40 km/h)
    ("low_mid", 6.0),
    ("low_small_up", 7.0),
    ("low_small_down", 5.0),
    ("low_bound_lo", 10.5),
    ("low_bound_hi", 11.5),
    ("low_bound_lo2", 10.5),
    # Mid range (40-80 km/h)
    ("mid_mid", 16.7),
    ("mid_small_up", 17.7),
    ("mid_small_down", 15.7),
    ("mid_bound_lo", 21.5),
    ("mid_bound_hi", 22.5),
    ("mid_bound_lo2", 21.5),
    # High range (80-120 km/h)
    ("high_mid", 27.8),
    ("high_small_up", 28.8),
    ("high_small_down", 26.8),
    # Larger steps crossing ranges
    ("big_back_to_0", 0.0),
    ("big_to_mid", 16.7),
    ("big_back_to_0_2", 0.0),
    ("big_to_high", 27.8),
    ("big_back_to_0_3", 0.0),
    # Pre-emergency cruise, then emergency is injected deterministically.
    ("cruise_for_emergency", EMERGENCY_TARGET_SP_MPS),
    ("_emergency_brake_event", EMERGENCY_TARGET_SP_MPS),
    ("post_emergency_stop", 0.0),
]


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
        super().__init__("following_ros_cam")

        self.args = args

        if args.plen < 1:
            raise ValueError("Platoon length (--plen) must be >= 1")

        # Sampling time / CARLA dt
        self.Ts = 0.01
        self.gate_on_first_throttle = bool(getattr(args, "gate_on_first_throttle", False))

        # Platoon configuration
        self.plen = int(args.plen)
        # Optional MCU-controlled vehicle index. If set, that vehicle uses the normal
        # indexed name (veh_<idx>) and no Python PlatoonMember controller is spawned for it.
        self.mcu_index = int(args.mcu_index) if args.mcu_index is not None else None
        if self.mcu_index is not None and not (0 <= self.mcu_index < self.plen):
            raise ValueError("--mcu-index must be in [0, plen-1]")

        # If no MCU is configured, do not gate the test waiting for an MCU throttle.
        # This avoids a deadlock where the global setpoint stays at 0 forever.
        if self.mcu_index is None and self.gate_on_first_throttle:
            self.get_logger().warn(
                "--gate-on-first-throttle ignored because no --mcu-index was provided; starting test immediately."
            )
            self.gate_on_first_throttle = False

        self.mcu_ros_name = f"veh_{self.mcu_index}" if self.mcu_index is not None else None

        # index of ego vehicle for logging/dashcam:
        # - if MCU index provided, use that as ego
        # - else last in platoon if >1, else leader
        if self.mcu_index is not None:
            self.ego_index = self.mcu_index
        else:
            self.ego_index = self.plen - 1 if self.plen > 1 else 0

        # Per-vehicle ROS publishers (filled after spawning vehicles)
        self.ros_names = []
        self.state_pubs = {}
        self.platoon_mode_pubs = {}
        self._mcu_setpoint_pub = None

        # QoS profiles: match MCU expectations (BEST_EFFORT + VOLATILE everywhere).
        self.qos_state = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.qos_setpoint = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.qos_cmd = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # Global platoon setpoint publisher
        self.platoon_sp_pub = self.create_publisher(
            Float32, "/platoon/plat_0/setpoint", self.qos_setpoint
        )

        # Latest commands from PlatoonMember nodes (name -> dict)
        self._last_cmd = {}
        # Wall-time of last received command messages (for latency analysis).
        # Values are seconds from time.perf_counter(); None means "never received".
        self._last_cmd_rx_wall_s = {}
        self._last_throttle_rx_wall_s = {}
        self._last_brake_rx_wall_s = {}
        self._last_applied_cmd_rx_wall_s = {}
        # Derived timing signals for plotting (per vehicle):
        # - inter-arrival times: dt between successive received messages
        # - applied age: how stale the last received cmd pair was when applied
        self._last_throttle_rx_dt_ms = {}
        self._last_brake_rx_dt_ms = {}
        self._last_cmd_pair_rx_wall_s = {}
        self._last_cmd_pair_rx_dt_ms = {}
        self._last_cmd_age_ms = {}

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
        base_spawn_point.location.y -= 100.0  # put the leader further ahead to let the followers have space to spawn
        self._spawn_points = spawn_points
        self._base_spawn_point = base_spawn_point

        vehicle_bp = bp_library.find("vehicle.tesla.model3")

        # Distance between vehicles at spawn so they don't collide immediately
        self.initial_spacing = 15.0

        self.vehicles = []
        
        self.prev_speeds = [] # Previous speed for exponential smoothing low pass filter
        
        self.alpha = 0.4

        # Spawn platoon members
        for i in range(self.plen):

            # Always use regular indexing for topic namespaces
            ros_name = f"veh_{i}"

            spawn_tf = carla.Transform(
                location=carla.Location(
                    x=base_spawn_point.location.x,
                    y=base_spawn_point.location.y + i * self.initial_spacing,
                    z=base_spawn_point.location.z,
                ),
                rotation=base_spawn_point.rotation,
            )

            if i == self.mcu_index:
                vehicle_bp.set_attribute('color', '255,0,0')
            else:
                vehicle_bp.set_attribute('color', '0,0,0')
                

            veh = world.spawn_actor(vehicle_bp, spawn_tf)
            veh.set_autopilot(False)

            self.get_logger().info(f"Spawned {veh.type_id} as {ros_name} at {spawn_tf}")

            self.ros_names.append(ros_name)
            self.vehicles.append(veh)
            self.prev_speeds.append(0.0)

            # First follower (second member) leaves more room as everyone receives a emergency stop
            # signal at the same time, but the leader must react itself before generating and sending
            # signal
            self._desired_time_headways[ros_name] = 1.0 if i == 1 else 0.50
            self._min_spacings[ros_name] = 8.0

            # State publishers (absolute topics)
            self.state_pubs[ros_name] = self.create_publisher(
                VehicleState, f"/{ros_name}/state", self.qos_state
            )
            self.platoon_mode_pubs[ros_name] = self.create_publisher(
                Bool, f"/{ros_name}/state/platoon_enabled", self.qos_setpoint
            )
            # Publish once (TRANSIENT_LOCAL) to avoid sending extra messages at 100 Hz.
            self.platoon_mode_pubs[ros_name].publish(Bool(data=True))

            # MCU expects <veh_idx>/state/setpoint (best effort). Only publish for MCU index.
            if self.mcu_index is not None and i == self.mcu_index:
                self._mcu_setpoint_pub = self.create_publisher(
                    Float32, f"/{ros_name}/state/setpoint", self.qos_state
                )

            # Initialize command/state caches for this vehicle.
            # Callbacks only accept messages for names present in these dicts.
            self._last_cmd[ros_name] = {"throttle": 0.0, "brake": 0.0}
            self._last_cmd_rx_wall_s[ros_name] = None
            self._last_throttle_rx_wall_s[ros_name] = None
            self._last_brake_rx_wall_s[ros_name] = None
            self._last_applied_cmd_rx_wall_s[ros_name] = None
            self._last_throttle_rx_dt_ms[ros_name] = None
            self._last_brake_rx_dt_ms[ros_name] = None
            self._last_cmd_pair_rx_wall_s[ros_name] = None
            self._last_cmd_pair_rx_dt_ms[ros_name] = None
            self._last_cmd_age_ms[ros_name] = None
            self._last_local_sp[ros_name] = 0.0

            # Command subscribers (absolute topics)
            self.create_subscription(
                Float32,
                f"/{ros_name}/command/throttle",
                lambda msg, n=ros_name: self._throttle_cb(msg, n),
                self.qos_cmd,
            )
            self.create_subscription(
                Float32,
                f"/{ros_name}/command/brake",
                lambda msg, n=ros_name: self._brake_cb(msg, n),
                self.qos_cmd,
            )

            # Local setpoint subscriber (absolute topic)
            self.create_subscription(
                Float32,
                f"/{ros_name}/state/local_setpoint",
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
            self.camera = world.spawn_actor(cam_bp, cam_tf, attach_to=self.vehicles[-1])
            self.get_logger().info(
                f"Spawned {self.camera.type_id} attached to {self.ros_names[-1]} (dashcam)"
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

        # Build header:
        # - time_s, teleport_event (0/1)
        # - standardized test plan state (global SP, stability, emergency flag)
        # - then for each vehicle a block of
        # throttle_<name>, brake_<name>, speed_<name>, accel_x_<name>,
        # local_sp_<name>, dist_to_prev_<name>, desired_dist_<name>, dist_err_<name>, vel_diff_<name>, ttc_<name>
        header = [
            "time_s",
            "teleport_event",
            "global_sp_mps",
            "plan_idx",
            "plan_name",
            "leader_speed_mps",
            "leader_in_band",
            "leader_stable_s",
            "emergency_active",
        ]
        for i, name in enumerate(self.ros_names):
            suffix = str(i)
            header.extend([
                f"throttle_{suffix}",
                f"brake_{suffix}",
                f"cmd_age_ms_{suffix}",
                f"throttle_rx_dt_ms_{suffix}",
                f"brake_rx_dt_ms_{suffix}",
                f"cmd_pair_rx_dt_ms_{suffix}",
                f"speed_{suffix}",
                f"accel_x_{suffix}",
                f"local_sp_{suffix}",
                f"dist_to_prev_{suffix}",
                f"desired_dist_{suffix}",
                f"dist_err_{suffix}",
                f"vel_diff_{suffix}",
                f"ttc_{suffix}",
            ])
        self.csv_writer.writerow(header)
        self.csv_file.flush()
        self.get_logger().info(f"Logging to: {csv_filename}")

        # Teleport event flag (set to 1 only on the step teleport occurs).
        self._teleport_event = 0

        # Step test definition (inspired by original platooning script)
        self.dt = settings.fixed_delta_seconds
        self.total_time = 3600.0
        self.total_steps = int(self.total_time / self.dt)

        # Distance-based teleport (matches following_python_test.py behavior)
        self.teleport_dist_m = float(args.teleport_dist)
        self.dist_since_tp_m = 0.0
        self.last_lead_loc = None

        # --- Standardized setpoint test plan (leader global setpoint) ---
        self._plan = []
        for name, sp_mps in list(DEFAULT_SETPOINT_PLAN):
            seg_type = "emergency" if str(name).startswith("_emergency") else "setpoint"
            self._plan.append({"type": seg_type, "name": str(name), "sp_mps": float(sp_mps)})

        self._plan_idx = 0
        self._segment_start_sim_t = 0.0
        self._leader_stable_steps = 0
        self._leader_in_band = 0
        self._leader_stable_s = 0.0
        self._global_sp_mps = float(self._plan[0]["sp_mps"]) if self._plan else 0.0

        # Emergency event state (deterministic leader brake hijack)
        self._emergency_active = False
        # What was actually applied in the last apply_controls() call (i.e. for the tick we just advanced).
        self._emergency_active_applied = False
        self._emergency_brake_end_s = None
        self._emergency_recovery_end_s = None

        self.finished = False

        # State for stability-based setpoint changes
        self.sim_time = 0.0
        self.start_timestamp = None
        self.step = 0

        # Optional: gate the setpoint test until we see the first throttle command for veh_ego.
        # Default OFF because it can deadlock if the MCU only publishes after seeing a nonzero setpoint.
        self._test_started = not self.gate_on_first_throttle
        self._test_started_step = None

        # Startup instrumentation (one-time logs)
        self._first_nonzero_sp_step = None
        self._first_ego_throttle_step = None
        self._first_ego_brake_step = None
        self._first_ego_cmd_applied_step = None
        self._last_mcu_age_log_step = None

        # Vehicle reset
        for v in self.vehicles:
            v.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # --- Optional leader random brake injection (test disturbances) ---
        # This is applied at the bridge level so it works whether the leader is
        # controlled by a Python PlatoonMember or by an MCU.
        self._leader_name = self.ros_names[0] if self.ros_names else "veh_0"
        self._leader_brake_enabled = bool(getattr(args, "leader_random_brakes", False))
        self._leader_brake_active = False
        self._leader_brake_cmd = 0.0
        self._leader_brake_end_s = 0.0
        self._leader_next_brake_s = float("inf")
        self._leader_rng = None

        self._leader_brake_min_interval_s = float(getattr(args, "leader_brake_min_interval", 20.0))
        self._leader_brake_max_interval_s = float(getattr(args, "leader_brake_max_interval", 60.0))
        if self._leader_brake_max_interval_s < self._leader_brake_min_interval_s:
            self._leader_brake_max_interval_s = self._leader_brake_min_interval_s
        self._leader_brake_duration_s = float(getattr(args, "leader_brake_duration", 0.75))
        self._leader_brake_min_cmd = float(getattr(args, "leader_brake_min_cmd", 0.05))
        self._leader_brake_max_cmd = float(getattr(args, "leader_brake_max_cmd", 0.15))
        if self._leader_brake_max_cmd < self._leader_brake_min_cmd:
            self._leader_brake_max_cmd = self._leader_brake_min_cmd
        self._leader_brake_min_speed_mps = float(getattr(args, "leader_brake_min_speed", 5.0))

        # Cache of last computed speeds (set in step_simulation) so the injector
        # doesn't need extra CARLA API calls.
        self._last_speeds = None

        if self._leader_brake_enabled:
            seed = getattr(args, "leader_brake_seed", None)
            if seed is None:
                # Use wall-clock time for variability.
                seed = int(time.time() * 1000.0) & 0xFFFFFFFF
            self._leader_rng = random.Random(int(seed))
            first_gap = self._leader_rng.uniform(self._leader_brake_min_interval_s, self._leader_brake_max_interval_s)
            self._leader_next_brake_s = float(first_gap)
            self.get_logger().info(
                "Leader random brakes ENABLED: "
                f"interval=[{self._leader_brake_min_interval_s:.1f},{self._leader_brake_max_interval_s:.1f}]s "
                f"duration={self._leader_brake_duration_s:.2f}s "
                f"cmd=[{self._leader_brake_min_cmd:.3f},{self._leader_brake_max_cmd:.3f}] "
                f"min_speed={self._leader_brake_min_speed_mps:.2f} m/s "
                f"seed={int(seed)}"
            )

        self.get_logger().info(
            f"Running platoon step test (plen={self.plen}): dt={self.dt}s, total_time={self.total_time}s, total_steps={self.total_steps}"
        )

    # --- Command callbacks from PlatoonMember nodes ---

    def _throttle_cb(self, msg: Float32, name: str) -> None:
        value = float(msg.data)
        if name in self._last_cmd:
            self._last_cmd[name]["throttle"] = value
            now_s = time.perf_counter()
            prev_s = self._last_throttle_rx_wall_s.get(name)
            self._last_throttle_rx_wall_s[name] = now_s
            self._last_cmd_rx_wall_s[name] = now_s
            if prev_s is not None:
                self._last_throttle_rx_dt_ms[name] = (float(now_s) - float(prev_s)) * 1000.0

            if self.mcu_ros_name is not None and name == self.mcu_ros_name and self._first_ego_throttle_step is None:
                self._first_ego_throttle_step = self.step
                self.get_logger().info(
                    f"First MCU throttle received ({name}) at step={self.step} sim_t={self.sim_time:.3f}s value={value:.6f}"
                )

            if self.mcu_ros_name is not None and name == self.mcu_ros_name and self.gate_on_first_throttle and not self._test_started:
                self._test_started = True
                self._test_started_step = self.step
                self.get_logger().info(
                    f"Test sequence enabled after first MCU throttle message ({name}) (step={self.step})."
                )
            self._update_cmd_pair_timing(name)

    def _brake_cb(self, msg: Float32, name: str) -> None:
        value = float(msg.data)
        if name in self._last_cmd:
            self._last_cmd[name]["brake"] = value
            now_s = time.perf_counter()
            prev_s = self._last_brake_rx_wall_s.get(name)
            self._last_brake_rx_wall_s[name] = now_s
            self._last_cmd_rx_wall_s[name] = now_s
            if prev_s is not None:
                self._last_brake_rx_dt_ms[name] = (float(now_s) - float(prev_s)) * 1000.0
            if self.mcu_ros_name is not None and name == self.mcu_ros_name and self._first_ego_brake_step is None:
                self._first_ego_brake_step = self.step
                self.get_logger().info(
                    f"First MCU brake received ({name}) at step={self.step} sim_t={self.sim_time:.3f}s value={value:.6f}"
                )
            self._update_cmd_pair_timing(name)

    def _latest_cmd_pair_rx_wall_s(self, name: str):
        """Return wall-time when a full (throttle+brake) pair was last received."""
        t_th = self._last_throttle_rx_wall_s.get(name)
        t_br = self._last_brake_rx_wall_s.get(name)
        if t_th is None or t_br is None:
            return None
        return max(float(t_th), float(t_br))

    def _update_cmd_pair_timing(self, name: str) -> None:
        """Track inter-arrival time of complete throttle+brake pairs."""
        pair_rx = self._latest_cmd_pair_rx_wall_s(name)
        if pair_rx is None:
            return
        prev_pair = self._last_cmd_pair_rx_wall_s.get(name)
        # Only update if the pair timestamp advances (can remain unchanged if the
        # "other" message is newer than the current one).
        if prev_pair is None or float(pair_rx) > float(prev_pair):
            if prev_pair is not None:
                self._last_cmd_pair_rx_dt_ms[name] = (float(pair_rx) - float(prev_pair)) * 1000.0
            self._last_cmd_pair_rx_wall_s[name] = float(pair_rx)

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
        # Reset event flag for this step; set to 1 only when teleport triggers below.
        self._teleport_event = 0

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
                self._teleport_event = 1
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
                        z=sp.location.z - 0.3,
                    )
                    v.set_transform(carla.Transform(location=new_loc, rotation=ref_rotation))

                self.dist_since_tp_m = 0.0
                self.last_lead_loc = self.vehicles[0].get_transform().location

        # Gather vehicle states
        speeds = []
        accels = []
        transforms = []
        idx = 0
        for v in self.vehicles:
            vel = v.get_velocity()
            accel = v.get_acceleration()
            tf = v.get_transform()
            # TODO low pass filter speed values, right now it is too noisy
            speed_raw = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            speed = self.alpha * speed_raw + ((1-self.alpha) * self.prev_speeds[idx])
            self.prev_speeds[idx] = speed
            idx += 1

            speeds.append(speed)
            accels.append(accel)
            transforms.append(tf)

        # Cache for use by the optional leader brake injector.
        self._last_speeds = list(speeds)

        # Current global leader speed setpoint (standardized test plan).
        # Optionally hold SP at 0 until first veh_ego throttle is received.
        leader_speed = float(speeds[0]) if speeds else 0.0
        plan_idx = int(self._plan_idx)
        plan_name = str(self._plan[plan_idx]["name"]) if self._plan else "none"

        if not self._test_started:
            sp = 0.0
            self._leader_in_band = 0
            self._leader_stable_steps = 0
            self._leader_stable_s = 0.0
        else:
            # Initialize segment timing when the plan effectively starts.
            if self._test_started_step is not None and self.step == int(self._test_started_step):
                self._segment_start_sim_t = float(self.sim_time)
                self._leader_stable_steps = 0
                self._leader_stable_s = 0.0

            seg = self._plan[plan_idx] if self._plan else {"type": "setpoint", "name": "none", "sp_mps": 0.0}
            sp = float(seg.get("sp_mps", 0.0))
            self._global_sp_mps = float(sp)

            seg_elapsed_s = float(self.sim_time) - float(self._segment_start_sim_t)
            in_band = 1 if abs(float(leader_speed) - float(sp)) <= float(STABLE_BAND_MPS) else 0
            self._leader_in_band = int(in_band)
            if in_band:
                self._leader_stable_steps += 1
            else:
                self._leader_stable_steps = 0
            self._leader_stable_s = float(self._leader_stable_steps) * float(self.dt)

            # Plan advancement logic
            advance = False
            reason = None
            if seg.get("type") == "setpoint":
                if (seg_elapsed_s >= float(MIN_SEGMENT_S)) and (self._leader_stable_s >= float(STABLE_HOLD_S)):
                    advance = True
                    reason = f"stable {self._leader_stable_s:.2f}s"
                elif seg_elapsed_s >= float(MAX_SEGMENT_S):
                    advance = True
                    reason = f"timeout {seg_elapsed_s:.2f}s"
            else:
                # Emergency segment: override leader brake for a fixed duration, then
                # allow a recovery window, while keeping the setpoint constant.
                if self._emergency_brake_end_s is None and self._emergency_recovery_end_s is None:
                    self._emergency_active = True
                    self._emergency_brake_end_s = float(self.sim_time) + float(EMERGENCY_BRAKE_DURATION_S)
                    self._emergency_recovery_end_s = float(self._emergency_brake_end_s) + float(EMERGENCY_RECOVERY_S)
                    self.get_logger().info(
                        f"EMERGENCY BRAKE START: t={self.sim_time:.2f}s cmd={float(EMERGENCY_BRAKE_CMD):.2f} "
                        f"duration={float(EMERGENCY_BRAKE_DURATION_S):.2f}s"
                    )

                if self._emergency_brake_end_s is not None and self._emergency_active and float(self.sim_time) >= float(self._emergency_brake_end_s):
                    self._emergency_active = False
                    self.get_logger().info(f"EMERGENCY BRAKE END: t={self.sim_time:.2f}s")

                if self._emergency_recovery_end_s is not None and float(self.sim_time) >= float(self._emergency_recovery_end_s):
                    advance = True
                    reason = "emergency complete"

            if advance and self._plan:
                prev_name = plan_name
                if self._plan_idx >= (len(self._plan) - 1):
                    self.finished = True
                    self.get_logger().info(f"Plan finished at t={self.sim_time:.2f}s (last segment={prev_name}).")
                else:
                    self._plan_idx += 1
                    plan_idx = int(self._plan_idx)
                    plan_name = str(self._plan[plan_idx]["name"])
                    self._segment_start_sim_t = float(self.sim_time)
                    self._leader_stable_steps = 0
                    self._leader_stable_s = 0.0
                    # Reset emergency state when leaving emergency segment.
                    self._emergency_active = False
                    self._emergency_brake_end_s = None
                    self._emergency_recovery_end_s = None
                    self.get_logger().info(
                        f"Plan advance: {prev_name} -> {plan_name} at t={self.sim_time:.2f}s (reason={reason})"
                    )
                    sp = float(self._plan[plan_idx].get("sp_mps", 0.0))
                    self._global_sp_mps = float(sp)

        if self._first_nonzero_sp_step is None and float(sp) > 1e-6:
            self._first_nonzero_sp_step = self.step
            self.get_logger().info(
                f"First nonzero platoon setpoint published at step={self.step} sim_t={self.sim_time:.3f}s sp={float(sp):.3f}"
            )

        self.platoon_sp_pub.publish(Float32(data=float(sp)))
        if self._mcu_setpoint_pub is not None:
            self._mcu_setpoint_pub.publish(Float32(data=float(sp)))

        # Publish per-vehicle state
        dists = []
        now_msg = self.get_clock().now().to_msg()
        for i, name in enumerate(self.ros_names):
            speed = speeds[i]

            if i == 0:
                dist_to_prev = 0.0
            else:
                prev_tf = transforms[i - 1]
                tf = transforms[i]
                dist_to_prev = math.dist(
                    (tf.location.x, tf.location.y, tf.location.z),
                    (prev_tf.location.x, prev_tf.location.y, prev_tf.location.z),
                )
            state_msg = VehicleState()
            state_msg.timestamp = now_msg
            state_msg.pos_x_m = float(transforms[i].location.x)
            state_msg.pos_y_m = float(transforms[i].location.y)
            state_msg.pos_z_m = float(transforms[i].location.z)
            state_msg.speed_mps = float(speed)
            state_msg.accel_mps2 = float(accels[i].x)
            state_msg.dist_to_prev = float(dist_to_prev)
            self.state_pubs[name].publish(state_msg)
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

        # Compute additional spacing signals for debugging/tuning:
        # - dist_err = dist_to_prev - desired_dist
        # - vel_diff = v_prev - v_ego
        # - ttc based on closing_speed = max(v_ego - v_prev, 0)
        dist_errs = []
        vel_diffs = []
        ttcs = []
        for i, name in enumerate(self.ros_names):
            if i == 0:
                dist_errs.append(0.0)
                vel_diffs.append(0.0)
                ttcs.append(float("inf"))
                continue

            dist_i = float(dists[i])
            desired_i = float(desired_dists[i])
            v_prev = float(speeds[i - 1])
            v_ego = float(speeds[i])

            dist_errs.append(dist_i - desired_i)
            vel_diffs.append(v_prev - v_ego)

            closing_speed = max(v_ego - v_prev, 0.0)
            if dist_i > 0.0 and closing_speed > 0.1:
                ttcs.append(dist_i / closing_speed)
            else:
                ttcs.append(float("inf"))

        # Note: plan advancement is handled above via leader stability gating.

        # Logging to CSV: per-vehicle throttle, brake, speed, accel_x,
        # local setpoint, distance to previous, desired distance, dist_err, vel_diff, ttc
        row = [
            f"{self.sim_time:.4f}",
            str(int(self._teleport_event)),
            f"{float(sp):.4f}",
            str(int(plan_idx)),
            str(plan_name),
            f"{float(leader_speed):.4f}",
            str(int(self._leader_in_band)),
            f"{float(self._leader_stable_s):.4f}",
            str(int(self._emergency_active_applied)),
        ]
        for i, name in enumerate(self.ros_names):
            cmd = self._last_cmd.get(name, {"throttle": 0.0, "brake": 0.0})
            throttle = float(cmd["throttle"])
            brake = float(cmd["brake"])
            cmd_age_ms = self._last_cmd_age_ms.get(name)
            th_dt_ms = self._last_throttle_rx_dt_ms.get(name)
            br_dt_ms = self._last_brake_rx_dt_ms.get(name)
            pair_dt_ms = self._last_cmd_pair_rx_dt_ms.get(name)
            speed_i = float(speeds[i])
            accel_x_i = float(accels[i].x)
            local_sp = float(self._last_local_sp.get(name, sp))
            dist_i = float(dists[i])
            desired_dist_i = float(desired_dists[i])
            dist_err_i = float(dist_errs[i])
            vel_diff_i = float(vel_diffs[i])
            ttc_i = float(ttcs[i])

            row.extend([
                f"{throttle:.4f}",
                f"{brake:.4f}",
                ("" if cmd_age_ms is None else f"{float(cmd_age_ms):.4f}"),
                ("" if th_dt_ms is None else f"{float(th_dt_ms):.4f}"),
                ("" if br_dt_ms is None else f"{float(br_dt_ms):.4f}"),
                ("" if pair_dt_ms is None else f"{float(pair_dt_ms):.4f}"),
                f"{speed_i:.4f}",
                f"{accel_x_i:.4f}",
                f"{local_sp:.4f}",
                f"{dist_i:.4f}",
                f"{desired_dist_i:.4f}",
                f"{dist_err_i:.4f}",
                f"{vel_diff_i:.4f}",
                f"{ttc_i:.4f}",
            ])

        self.csv_writer.writerow(row)
        self.csv_file.flush()

        self.step += 1

    def apply_controls(self) -> None:
        """Apply latest commands from controllers to CARLA vehicles.

        This should be called exactly once before each world.tick(), so that
        the commands computed on the previous tick are used for the next
        simulation step.
        """
        # Capture the emergency flag for the controls we are about to apply.
        self._emergency_active_applied = bool(getattr(self, "_emergency_active", False))

        for name, v in zip(self.ros_names, self.vehicles):
            cmd = self._last_cmd.get(name, {"throttle": 0.0, "brake": 0.0})
            throttle_cmd = max(0.0, min(1.0, float(cmd["throttle"])))
            brake_cmd = max(0.0, min(1.0, float(cmd["brake"])))

            # Deterministic emergency brake hijack on the leader (used by the
            # standardized test plan). This overrides the controller output.
            if name == self._leader_name and bool(self._emergency_active_applied):
                throttle_cmd = 0.0
                brake_cmd = max(float(brake_cmd), float(EMERGENCY_BRAKE_CMD))
                try:
                    self._last_cmd[name]["throttle"] = float(throttle_cmd)
                    self._last_cmd[name]["brake"] = float(brake_cmd)
                except Exception:
                    pass

            # Optional: inject mild random brake pulses on the leader.
            if (
                self._leader_brake_enabled
                and name == self._leader_name
                and self._leader_rng is not None
                and (not bool(self._emergency_active_applied))
            ):
                sim_t = float(self.sim_time)
                leader_speed = None
                try:
                    if self._last_speeds is not None and len(self._last_speeds) > 0:
                        leader_speed = float(self._last_speeds[0])
                except Exception:
                    leader_speed = None

                # End an active pulse.
                if self._leader_brake_active and sim_t >= float(self._leader_brake_end_s):
                    self._leader_brake_active = False
                    self._leader_brake_cmd = 0.0
                    gap = self._leader_rng.uniform(
                        float(self._leader_brake_min_interval_s),
                        float(self._leader_brake_max_interval_s),
                    )
                    self._leader_next_brake_s = sim_t + float(gap)
                    self.get_logger().info(
                        f"Leader random brake END at t={sim_t:.2f}s; next in {gap:.1f}s"
                    )

                # Start a new pulse if conditions are met.
                if (
                    (not self._leader_brake_active)
                    and self._test_started
                    and sim_t >= float(self._leader_next_brake_s)
                    and (leader_speed is None or leader_speed >= float(self._leader_brake_min_speed_mps))
                ):
                    self._leader_brake_active = True
                    self._leader_brake_cmd = float(
                        self._leader_rng.uniform(
                            float(self._leader_brake_min_cmd),
                            float(self._leader_brake_max_cmd),
                        )
                    )
                    self._leader_brake_end_s = sim_t + float(self._leader_brake_duration_s)
                    self.get_logger().info(
                        f"Leader random brake START at t={sim_t:.2f}s cmd={self._leader_brake_cmd:.3f} "
                        f"duration={self._leader_brake_duration_s:.2f}s"
                    )

                if self._leader_brake_active:
                    throttle_cmd = 0.0
                    brake_cmd = max(float(brake_cmd), float(self._leader_brake_cmd))
                    # Keep logging consistent with what's actually applied.
                    try:
                        self._last_cmd[name]["throttle"] = float(throttle_cmd)
                        self._last_cmd[name]["brake"] = float(brake_cmd)
                    except Exception:
                        pass

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
            cmd_pair_rx_s = self._latest_cmd_pair_rx_wall_s(name)
            if cmd_pair_rx_s is not None:
                self._last_applied_cmd_rx_wall_s[name] = cmd_pair_rx_s
                self._last_cmd_age_ms[name] = (time.perf_counter() - float(cmd_pair_rx_s)) * 1000.0
            else:
                self._last_cmd_age_ms[name] = None
            if (
                self.mcu_ros_name is not None
                and name == self.mcu_ros_name
                and (self.step % 50) == 0
            ):
                if self._last_mcu_age_log_step != self.step:
                    self._last_mcu_age_log_step = self.step
                    if cmd_pair_rx_s is None:
                        self.get_logger().info("MCU cmd age: N/A (no throttle+brake received yet)")
                    else:
                        age_ms = (time.perf_counter() - float(cmd_pair_rx_s)) * 1000.0
                        th_dt = self._last_throttle_rx_dt_ms.get(name)
                        br_dt = self._last_brake_rx_dt_ms.get(name)
                        pair_dt = self._last_cmd_pair_rx_dt_ms.get(name)
                        self.get_logger().info(
                            "MCU cmd age at apply: "
                            f"{age_ms:.2f} ms "
                            f"(th_dt={('N/A' if th_dt is None else f'{float(th_dt):.2f}')} ms, "
                            f"br_dt={('N/A' if br_dt is None else f'{float(br_dt):.2f}')} ms, "
                            f"pair_dt={('N/A' if pair_dt is None else f'{float(pair_dt):.2f}')} ms)"
                        )

            if (
                self.mcu_ros_name is not None
                and name == self.mcu_ros_name
                and self._first_ego_cmd_applied_step is None
                and (abs(throttle_cmd) > 1e-6 or abs(brake_cmd) > 1e-6)
            ):
                self._first_ego_cmd_applied_step = self.step
                self.get_logger().info(
                    f"First nonzero MCU control applied ({name}) at step={self.step} sim_t={self.sim_time:.3f}s "
                    f"throttle={throttle_cmd:.3f} brake={brake_cmd:.3f}"
                )

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
        "--mcu-index",
        type=int,
        default=None,
        help="Optional index [0..plen-1] of the MCU vehicle (uses normal name veh_<idx>) and skips the Python controller for it.",
    )
    parser.add_argument(
        "--teleport-dist",
        type=float,
        default=320.0,
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
    parser.add_argument(
        "--wall-dt",
        type=float,
        default=None,
        help="Optional wall-clock period per CARL0 dist=10.01 sp=0.00 th=0.00 br=0.01 ff=0 ttc=0 dist=0 over=0 des_dec=0.00 cam_dt=0.103s cam_dt_wall=0.103s                                    A tick [s]. If unset, uses fixed_delta_seconds.",
    )
    # ACC tuning knobs (local setpoint shaping)
    parser.add_argument("--k-dist", type=float, default=0.8, help="Distance-error gain for local setpoint (normal).")
    parser.add_argument("--k-vel", type=float, default=0.0, help="Relative-speed gain for local setpoint (normal).")
    parser.add_argument(
        "--k-dist-panic",
        type=float,
        default=0.8,
        help="Distance-error gain in panic mode (defaults to --k-dist).",
    )
    parser.add_argument(
        "--k-vel-panic",
        type=float,
        default=0.0,
        help="Relative-speed gain in panic mode (defaults to --k-vel).",
    )
    parser.add_argument(
        "--ttc-panic-s",
        type=float,
        default=2.0,
        help="Enter panic mode when TTC to predecessor is <= this threshold (seconds).",
    )
    parser.add_argument(
        "--dist-err-deadband-m",
        type=float,
        default=0.0,
        help="Deadband on distance error (meters) before it affects local setpoint.",
    )
    parser.add_argument(
        "--vel-diff-deadband-mps",
        type=float,
        default=0.0,
        help="Deadband on relative speed (m/s) before it affects local setpoint.",
    )
    parser.add_argument(
        "--gate-on-first-throttle",
        action="store_true",
        help="Hold platoon setpoint at 0 until first MCU throttle is received (ignored if --mcu-index is not set).",
    )

    # --- Leader disturbance injection: mild random brakes ---
    parser.add_argument(
        "--leader-random-brakes",
        action="store_true",
        help="Inject infrequent mild brake pulses on the leader (veh_0) to test follower response.",
    )
    parser.add_argument(
        "--leader-brake-seed",
        type=int,
        default=None,
        help="Random seed for leader brake events (default: derived from wall-clock time).",
    )
    parser.add_argument(
        "--leader-brake-min-interval",
        type=float,
        default=30.0,
        help="Minimum time between leader brake pulses [s] (sim time).",
    )
    parser.add_argument(
        "--leader-brake-max-interval",
        type=float,
        default=60.0,
        help="Maximum time between leader brake pulses [s] (sim time).",
    )
    parser.add_argument(
        "--leader-brake-duration",
        type=float,
        default=0.75,
        help="Duration of each leader brake pulse [s] (sim time).",
    )
    parser.add_argument(
        "--leader-brake-min-cmd",
        type=float,
        default=0.8,
        help="Minimum brake command for leader pulses [0..1].",
    )
    parser.add_argument(
        "--leader-brake-max-cmd",
        type=float,
        default=1.0,
        help="Maximum brake command for leader pulses [0..1].",
    )
    parser.add_argument(
        "--leader-brake-min-speed",
        type=float,
        default=5.0,
        help="Only inject leader brake pulses when leader speed is at least this value [m/s].",
    )
    args = parser.parse_args()

    rclpy.init()
    bridge = FollowingRosTest(args)

    # Create Python ROS platoon controller nodes (PlatoonMember) for all vehicles
    Ts = bridge.Ts

    # PID scheduling gains (match following_python_test.py / Platooning.py)
    kb_aw = 1.0
    u_min, u_max = -1.0, 1.0 # TODO Bound to 0.0 to 1.0 and let brake arbiter deal with braking

    nodes = [bridge]
    for i, name in enumerate(bridge.ros_names):
        if bridge.mcu_index is not None and i == bridge.mcu_index:
            continue
        platoon_index = i

        # Outdated values
        #slow_pid = PID(0.43127789, 0.43676547, 0.0, 15.0, Ts, u_min, u_max, kb_aw, der_on_meas=True)
        #mid_pid = PID(0.11675119, 0.085938,   0.0, 14.90530836, Ts, u_min, u_max, kb_aw, der_on_meas=True)
        #fast_pid = PID(0.13408096, 0.07281374, 0.0, 12.16810135, Ts, u_min, u_max, kb_aw, der_on_meas=True)

        # New values, must check in sim
        slow_pid = PID(0.1089194,0.04906409, 0.0, 7.71822214, Ts, u_min, u_max, kb_aw, der_on_meas=True)
        mid_pid = PID(0.11494122, 0.0489494,  0.0, 5.0 , Ts, u_min, u_max, kb_aw, der_on_meas=True)
        fast_pid = PID( 0.19021246, 0.15704399, 0.0, 12.70886655, Ts, u_min, u_max, kb_aw, der_on_meas=True)

        # Use the same spacing parameters as stored in the bridge so that the
        # desired distance used by controllers matches what is logged.
        desired_time_headway = float(bridge._desired_time_headways.get(name, 0.3))
        min_spacing = float(bridge._min_spacings.get(name, 5.0))

        member = PlatoonMember(
            name=name,
            platoon_index=platoon_index,
            slow_pid=slow_pid,
            mid_pid=mid_pid,
            fast_pid=fast_pid,
            desired_time_headway=desired_time_headway,
            min_spacing=min_spacing,
            K_dist=float(args.k_dist),
            K_vel=float(args.k_vel),
            K_dist_panic=(None if args.k_dist_panic is None else float(args.k_dist_panic)),
            K_vel_panic=(None if args.k_vel_panic is None else float(args.k_vel_panic)),
            ttc_panic_s=float(args.ttc_panic_s),
            dist_err_deadband_m=float(args.dist_err_deadband_m),
            vel_diff_deadband_mps=float(args.vel_diff_deadband_mps),
            K_brake=0.7,
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

    # Main simulation loop:
    # - CARLA is synchronous with fixed_dt = 0.01s sim time.
    # - Wall-time tick rate is capped to <= 1/dt to avoid overrunning the MCU.
    # - Between ticks, we keep spinning ROS so serial/micro-ROS traffic is handled
    #   promptly (reduces command latency vs. sleep-based pacing).
    dt_wall_s = float(bridge.dt) if args.wall_dt is None else float(args.wall_dt)
    state_delivery_budget_s = min(0.002, dt_wall_s * 0.25)

    # Prime with a safe initial control before first tick.
    bridge.apply_controls()
    tick_due_wall_s = time.perf_counter()
    try:
        while not bridge.finished:
            # Spin until the next tick boundary, but keep servicing ROS callbacks.
            while True:
                now_s = time.perf_counter()
                remaining_s = tick_due_wall_s - now_s
                if remaining_s <= 0.0:
                    break
                executor.spin_once(timeout_sec=remaining_s)

            tick_start_wall_s = time.perf_counter()

            # Apply the latest known controls just before advancing CARLA.
            bridge.apply_controls()

            # Advance CARLA one synchronous step and publish state.
            bridge.step_simulation()

            # Deliver state and setpoint messages to PlatoonMember nodes promptly.
            drain_executor(max_callbacks=500, max_wall_s=state_delivery_budget_s)

            # Run one control update per PlatoonMember (deterministic per-tick).
            for n in nodes:
                if isinstance(n, PlatoonMember):
                    n.step_once()

            # From here until the next tick boundary, keep spinning so:
            # - Python controllers' command publishes reach the bridge subscribers
            # - MCU (serial) command messages are received with minimal delay
            tick_due_wall_s = tick_start_wall_s + dt_wall_s
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
