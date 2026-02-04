"""
CARLA/ROS bridge for the cascaded controller (speed PID + accel PID).

What it does
  - Spawns a platoon in CARLA and publishes VehicleState topics.
  - Runs PlatooningROSCAMAccelPID nodes and applies their commands in CARLA.
  - Supports lane-keeping, dashcam, and CSV logging.

Outputs
  - CSV logs with per-vehicle state and control signals.

Run (example)
  python3 lib/following_ros_cam_accelpid.py --host localhost --port 2000 --plen 3
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
from lib.PlatooningROSCAMAccelPID import PlatoonMember


def closest_spawn_point(spawn_points, loc: carla.Location) -> carla.Transform:
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


class FollowingRosAccelPidBridge(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("following_ros_cam_accelpid")

        self.args = args
        if args.plen < 1:
            raise ValueError("Platoon length (--plen) must be >= 1")

        self.Ts = 0.01
        self.plen = int(args.plen)
        self.mcu_index = int(args.mcu_index) if args.mcu_index is not None else None
        if self.mcu_index is not None and not (0 <= self.mcu_index < self.plen):
            raise ValueError("--mcu-index must be in [0, plen-1]")

        self.ego_index = self.mcu_index if self.mcu_index is not None else (self.plen - 1 if self.plen > 1 else 0)

        self.ros_names = []
        self.state_pubs = {}
        self.platoon_mode_pubs = {}
        self._mcu_setpoint_pub = None

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

        self.platoon_sp_pub = self.create_publisher(
            Float32, "/platoon/plat_0/setpoint", self.qos_setpoint
        )

        self._last_cmd = {}
        self._last_local_sp = {}
        self._last_dbg = {}

        # spacing params mirrored from original bridge
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

        self.original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.Ts
        settings.no_rendering_mode = False
        settings.substepping = True
        settings.max_substep_delta_time = 0.001
        settings.max_substeps = 10 
        world.apply_settings(settings)
        self.dt = float(self.Ts)
        self.get_logger().info(f"Synchronous mode ON, dt = {self.Ts} s")

        spawn_points = world.get_map().get_spawn_points()
        base_spawn_point = spawn_points[125]
        base_spawn_point.location.y -= 60.0
        self._spawn_points = spawn_points
        self._base_spawn_point = base_spawn_point

        vehicle_bp = bp_library.find("vehicle.tesla.model3")
        self.initial_spacing = 15.0

        self.vehicles = []
        for i in range(self.plen):
            ros_name = f"veh_{i}"
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

            self._desired_time_headways[ros_name] = 0.3 if i == 1 else 0.10
            self._min_spacings[ros_name] = 4.0

            self.state_pubs[ros_name] = self.create_publisher(
                VehicleState, f"/{ros_name}/state", self.qos_state
            )
            self.platoon_mode_pubs[ros_name] = self.create_publisher(
                Bool, f"/{ros_name}/state/platoon_enabled", self.qos_setpoint
            )

            if self.mcu_index is not None and i == self.mcu_index:
                self._mcu_setpoint_pub = self.create_publisher(
                    Float32, f"/{ros_name}/state/setpoint", self.qos_state
                )

            self._last_cmd[ros_name] = {"throttle": 0.0, "brake": 0.0}
            self._last_dbg[ros_name] = {"a_des": 0.0, "a_err": 0.0, "v_err": 0.0}

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
            self.create_subscription(
                Float32,
                f"/{ros_name}/state/local_setpoint",
                lambda msg, n=ros_name: self._local_sp_cb(msg, n),
                self.qos_setpoint,
            )
            self.create_subscription(
                Float32,
                f"/{ros_name}/debug/a_des",
                lambda msg, n=ros_name: self._dbg_cb(msg, n, "a_des"),
                self.qos_cmd,
            )
            self.create_subscription(
                Float32,
                f"/{ros_name}/debug/a_err",
                lambda msg, n=ros_name: self._dbg_cb(msg, n, "a_err"),
                self.qos_cmd,
            )
            self.create_subscription(
                Float32,
                f"/{ros_name}/debug/v_err",
                lambda msg, n=ros_name: self._dbg_cb(msg, n, "v_err"),
                self.qos_cmd,
            )

        # Dashcam infrastructure (match following_ros_cam.py behavior by default)
        self._img_queue = None
        self._dashcam_stop = threading.Event()
        self._dashcam_thread = None
        self.camera = None
        if not bool(getattr(args, "no_dashcam", False)):
            try:
                self._img_queue = queue.Queue(maxsize=1)

                cam_bp = bp_library.find("sensor.camera.rgb")
                cam_bp.set_attribute("image_size_x", "640")
                cam_bp.set_attribute("image_size_y", "480")
                cam_bp.set_attribute("fov", "90")

                cam_tf = carla.Transform(
                    carla.Location(x=0.0, y=10.0, z=10.0),
                    carla.Rotation(yaw=-20.0, pitch=-35.0),
                )
                ego_vehicle = self.vehicles[self.ego_index]
                self.camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego_vehicle)
                self.get_logger().info(
                    f"Spawned {self.camera.type_id} attached to {self.ros_names[self.ego_index]} (dashcam)"
                )
                self.camera.listen(self._camera_callback)

                self._dashcam_thread = threading.Thread(
                    target=self._dashcam_worker, name="dashcam_viewer", daemon=True
                )
                self._dashcam_thread.start()
            except Exception as e:
                self.get_logger().warn(f"Failed to create dashcam: {e}")

        # Logging
        self.csv_file = open(args.f, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self._extra_csv_fields = list(getattr(args, "extra_csv_fields", []) or [])
        self._extra_csv_values = {k: "" for k in self._extra_csv_fields}
        header = ["time_s", "teleport_event"]
        header.extend([f"{k}" for k in self._extra_csv_fields])
        for name in self.ros_names:
            suffix = name
            header.extend(
                [
                    f"throttle_{suffix}",
                    f"brake_{suffix}",
                    f"speed_{suffix}",
                    f"speed_long_{suffix}",
                    f"accel_long_{suffix}",
                    f"local_sp_{suffix}",
                ]
            )
        self.csv_writer.writerow(header)
        self.csv_file.flush()

        # Setpoint sequence (same logic as following_ros_cam.py)
        self.lead_speed_setpoints = [
            10.0, 
            15.0, 
            20.0,
            33.0,
            10.0,
        ]
        self.lead_sp_idx = 0
        self.band = 0.2
        self.min_wait_steps = int(4.0 / self.Ts)
        self.last_change_step = 0
        self.last_change_speed = 0.0

        self.step = 0
        self.sim_time = 0.0
        self.start_timestamp = None
        self.finished = False

        # Acceleration estimation (match accel_fitting_data.py: multi-step speed diff + LPF)
        # Unify with tuning scripts: use longitudinal speed (dot with forward)
        # and a LPF of a_raw = d(v_long)/dt.
        self._accel_alpha = float(getattr(args, "accel_lpf_alpha", 0.35))
        self._v_long_prev = [0.0 for _ in range(self.plen)]
        self._accel_filts = [0.0 for _ in range(self.plen)]
        # Public-ish snapshots used by testbench scripts
        self._last_speeds = [0.0 for _ in range(self.plen)]         # speed magnitude (planar)
        self._last_speeds_long = [0.0 for _ in range(self.plen)]    # signed longitudinal speed
        self._last_accels_long = [0.0 for _ in range(self.plen)]    # longitudinal accel (LPF of dv_long/dt)

        # Teleport distance-based (same as original)
        self.teleport_dist_m = float(getattr(args, "teleport_dist", 320.0))
        self.dist_since_tp_m = 0.0
        self.last_lead_loc = None
        self._teleport_event = 0

    def _camera_callback(self, img) -> None:
        if self._img_queue is None:
            return
        payload = (bytes(img.raw_data), int(img.width), int(img.height))
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
        if self._img_queue is None:
            return
        time.sleep(1.0)
        try:
            cv2.namedWindow("Following ROS dashcam", cv2.WINDOW_NORMAL)
        except Exception as e:
            self.get_logger().warn(f"Failed to create OpenCV window: {e}")
            return

        try:
            while not self._dashcam_stop.is_set():
                try:
                    payload = self._img_queue.get(timeout=0.05)
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

    def _throttle_cb(self, msg: Float32, name: str) -> None:
        self._last_cmd[name]["throttle"] = float(msg.data)

    def _brake_cb(self, msg: Float32, name: str) -> None:
        self._last_cmd[name]["brake"] = float(msg.data)

    def _local_sp_cb(self, msg: Float32, name: str) -> None:
        self._last_local_sp[name] = float(msg.data)

    def _dbg_cb(self, msg: Float32, name: str, key: str) -> None:
        try:
            self._last_dbg[name][key] = float(msg.data)
        except Exception:
            pass

    def apply_controls(self) -> None:
        for name, v in zip(self.ros_names, self.vehicles):
            cmd = self._last_cmd.get(name, {"throttle": 0.0, "brake": 0.0})
            throttle_cmd = max(0.0, min(1.0, float(cmd["throttle"])))
            brake_cmd = max(0.0, min(1.0, float(cmd["brake"])))

            steer_cmd = 0.0
            if bool(getattr(self.args, "lane_keep", False)):
                steer_cmd = lane_keep_steer(
                    v,
                    self.carla_map,
                    lookahead_m=float(getattr(self.args, "lane_lookahead", 12.0)),
                    steer_gain=float(getattr(self.args, "lane_gain", 1.0)),
                )

            v.apply_control(
                carla.VehicleControl(
                    throttle=throttle_cmd,
                    brake=brake_cmd,
                    steer=float(steer_cmd),
                    hand_brake=False,
                    reverse=False,
                )
            )

    def step_simulation(self) -> None:
        self._teleport_event = 0

        self.world.tick()
        snapshot = self.world.get_snapshot()
        t = snapshot.timestamp.elapsed_seconds
        if self.start_timestamp is None:
            self.start_timestamp = t
        self.sim_time = t - self.start_timestamp

        # Teleport based on distance traveled by leader
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
                # Reset accel estimator state to avoid teleport-induced artifacts.
                self._v_long_prev = [0.0 for _ in range(self.plen)]
                self._accel_filts = [0.0 for _ in range(self.plen)]

        # Gather states and publish VehicleState
        speeds_abs = []
        speeds_long = []
        accels_long = []
        transforms = []
        for v in self.vehicles:
            vel = v.get_velocity()
            tf = v.get_transform()

            # Planar speed magnitude (legacy / logging)
            speed_abs = math.sqrt(vel.x ** 2 + vel.y ** 2)
            speeds_abs.append(speed_abs)
            transforms.append(tf)

        # Longitudinal speed (signed) and accel estimation: a_raw = d(v_long)/dt then LPF
        for i, v in enumerate(self.vehicles):
            vel = v.get_velocity()
            fwd = v.get_transform().get_forward_vector()
            v_long = float(vel.x * fwd.x + vel.y * fwd.y + vel.z * fwd.z)
            speeds_long.append(v_long)

            a_raw = (v_long - float(self._v_long_prev[i])) / float(self.Ts)
            self._v_long_prev[i] = v_long

            a_f = float(self._accel_filts[i])
            alpha = float(self._accel_alpha)
            a_f = alpha * float(a_raw) + (1.0 - alpha) * a_f
            self._accel_filts[i] = a_f
            accels_long.append(a_f)

        self._last_speeds = list(speeds_abs)
        self._last_speeds_long = list(speeds_long)
        self._last_accels_long = list(accels_long)

        for i, name in enumerate(self.ros_names):
            dist_to_prev = 0.0
            if i > 0:
                dist_to_prev = transforms[i].location.distance(transforms[i - 1].location)

            state_msg = VehicleState()
            state_msg.timestamp = self.get_clock().now().to_msg()
            state_msg.pos_x_m = float(transforms[i].location.x)
            state_msg.pos_y_m = float(transforms[i].location.y)
            state_msg.pos_z_m = float(transforms[i].location.z)
            # Keep message semantics as "speed magnitude", but expose longitudinal speed via the internal lists.
            state_msg.speed_mps = float(speeds_abs[i])
            state_msg.accel_mps2 = float(accels_long[i])
            state_msg.dist_to_prev = float(dist_to_prev)
            self.state_pubs[name].publish(state_msg)

            # For completeness (MCU optional) publish base setpoint too.
            if self._mcu_setpoint_pub is not None and i == self.mcu_index:
                self._mcu_setpoint_pub.publish(Float32(data=float(self.lead_speed_setpoints[self.lead_sp_idx])))

            self.platoon_mode_pubs[name].publish(Bool(data=True))

        # Step logic to change lead SP when ego settles (matches original script)
        if self.plen > 1 and (self.step - self.last_change_step) >= self.min_wait_steps:
            ego_speed = float(speeds_abs[self.ego_index])
            if abs(ego_speed - self.last_change_speed) <= self.band:
                if self.lead_sp_idx == (len(self.lead_speed_setpoints) - 1):
                    self.finished = True
                else:
                    self.lead_sp_idx = (self.lead_sp_idx + 1) % len(self.lead_speed_setpoints)
                self.get_logger().info(
                    f"Lead SP change to {self.lead_speed_setpoints[self.lead_sp_idx]:.3f} m/s at t={self.sim_time:.2f}s"
                )
                self.last_change_step = self.step
                self.last_change_speed = ego_speed
            else:
                self.last_change_step = self.step
                self.last_change_speed = ego_speed

        sp = float(self.lead_speed_setpoints[self.lead_sp_idx])
        self.platoon_sp_pub.publish(Float32(data=sp))

        # Log
        row = [f"{self.sim_time:.4f}", str(int(self._teleport_event))]
        for k in self._extra_csv_fields:
            row.append(str(self._extra_csv_values.get(k, "")))
        for i, name in enumerate(self.ros_names):
            cmd = self._last_cmd.get(name, {"throttle": 0.0, "brake": 0.0})
            row.extend(
                [
                    f"{float(cmd['throttle']):.4f}",
                    f"{float(cmd['brake']):.4f}",
                    f"{float(speeds_abs[i]):.4f}",
                    f"{float(speeds_long[i]):.4f}",
                    f"{float(accels_long[i]):.4f}",
                    f"{float(self._last_local_sp.get(name, sp)):.4f}",
                ]
            )
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        self.step += 1

    def set_extra_csv_values(self, values: dict) -> None:
        """Optional: provide extra per-tick values that will be logged in the CSV."""
        if not hasattr(self, "_extra_csv_fields"):
            return
        for k in self._extra_csv_fields:
            if k in values:
                self._extra_csv_values[k] = values[k]

    def reset_test(self) -> None:
        """Teleport vehicles back, force stop, and restart the SP sequence."""
        try:
            self.get_logger().info("=== RESET: teleporting + stopping vehicles ===")
        except Exception:
            pass

        try:
            base_tf = self._base_spawn_point
            ref_rotation = base_tf.rotation
            for i, v in enumerate(self.vehicles):
                target_loc = carla.Location(
                    x=base_tf.location.x,
                    y=base_tf.location.y + i * self.initial_spacing,
                    z=base_tf.location.z,
                )
                sp = closest_spawn_point(self._spawn_points, target_loc)
                new_loc = carla.Location(
                    x=target_loc.x,
                    y=target_loc.y,
                    z=(sp.location.z - 0.3) if sp is not None else target_loc.z,
                )
                v.set_transform(carla.Transform(location=new_loc, rotation=ref_rotation))
                v.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                v.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                v.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
        except Exception as e:
            try:
                self.get_logger().warn(f"Reset teleport failed: {e}")
            except Exception:
                pass

        # Reset command cache so the next tick starts from a safe command.
        for name in self.ros_names:
            self._last_cmd[name] = {"throttle": 0.0, "brake": 0.0}

        # Reset setpoint sequencing
        self.lead_sp_idx = 0
        self.last_change_step = self.step
        self.last_change_speed = 0.0

        # Reset teleport/accel estimator state
        self.dist_since_tp_m = 0.0
        self.last_lead_loc = None
        self._speed_histories = [[] for _ in range(self.plen)]
        self._accel_filts = [0.0 for _ in range(self.plen)]

    def shutdown(self) -> None:
        self.get_logger().info("Stopping vehicles and cleaning up...")
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
        except Exception:
            pass
        try:
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
    parser = argparse.ArgumentParser(description="CARLA platooning (accelPID) bridge (ROS2)")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("-f", type=str, default="out_ros_accelpid.csv")
    parser.add_argument("--plen", type=int, default=2)
    parser.add_argument("--mcu-index", type=int, default=None)
    parser.add_argument("--teleport-dist", type=float, default=320.0)
    parser.add_argument("--no-dashcam", action="store_true")
    parser.add_argument("--lane-keep", action="store_true")
    parser.add_argument("--lane-lookahead", type=float, default=12.0)
    parser.add_argument("--lane-gain", type=float, default=1.0)
    parser.add_argument("--wall-dt", type=float, default=None)
    parser.add_argument(
        "--print-every-s",
        type=float,
        default=1.0,
        help="Print a live debug line every N seconds (sim time). Set 0 to disable.",
    )
    parser.add_argument("--max-sim-time-s", type=float, default=0.0, help="Stop after this sim time (0 = run until Ctrl+C).")

    # Tuned gains (single set applied to all speed ranges).
    parser.add_argument("--speed-kp", type=float, default=0.7)
    parser.add_argument("--speed-ki", type=float, default=0.05)
    parser.add_argument("--speed-kd", type=float, default=0.05)
    parser.add_argument("--speed-n", type=float, default=12.0)
    parser.add_argument("--accel-kp", type=float, default=1.0)
    parser.add_argument("--accel-ki", type=float, default=0.10)
    parser.add_argument("--accel-kd", type=float, default=0.05)
    parser.add_argument("--accel-n", type=float, default=12.0)

    # Limits
    parser.add_argument("--a-min", type=float, default=-6.0, help="Outer PID output min (m/s^2).")
    parser.add_argument("--a-max", type=float, default=3.0, help="Outer PID output max (m/s^2).")

    # ACC tuning knobs (local setpoint shaping)
    parser.add_argument("--k-dist", type=float, default=2.0, help="Distance-error gain for local setpoint (normal).")
    parser.add_argument("--k-vel", type=float, default=1.0, help="Relative-speed gain for local setpoint (normal).")
    parser.add_argument(
        "--k-dist-panic",
        type=float,
        default=None,
        help="Distance-error gain in panic mode (defaults to --k-dist).",
    )
    parser.add_argument(
        "--k-vel-panic",
        type=float,
        default=None,
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
    args = parser.parse_args()

    rclpy.init()
    bridge = FollowingRosAccelPidBridge(args)

    Ts = bridge.Ts
    kb_aw = 1.0

    # Outer speed->accel PID output limits; inner accel PID outputs [-1..1].
    a_min, a_max = float(args.a_min), float(args.a_max)
    u_min, u_max = -1.0, 1.0

    nodes = [bridge]
    for i, name in enumerate(bridge.ros_names):
        if bridge.mcu_index is not None and i == bridge.mcu_index:
            continue

        # Per-vehicle PID instances (do not share state between vehicles)
        speed_pid_low = PID(
            float(args.speed_kp),
            float(args.speed_ki),
            float(args.speed_kd),
            float(args.speed_n),
            Ts,
            a_min,
            a_max,
            kb_aw,
            der_on_meas=True,
        )
        speed_pid_mid = PID(
            float(args.speed_kp),
            float(args.speed_ki),
            float(args.speed_kd),
            float(args.speed_n),
            Ts,
            a_min,
            a_max,
            kb_aw,
            der_on_meas=True,
        )
        speed_pid_high = PID(
            float(args.speed_kp),
            float(args.speed_ki),
            float(args.speed_kd),
            float(args.speed_n),
            Ts,
            a_min,
            a_max,
            kb_aw,
            der_on_meas=True,
        )

        # Inner accel->effort PIDs (scheduled, outputs [-1, 1])
        accel_pid_low = PID(
            float(args.accel_kp),
            float(args.accel_ki),
            float(args.accel_kd),
            float(args.accel_n),
            Ts,
            u_min,
            u_max,
            kb_aw,
            der_on_meas=True,
        )
        accel_pid_mid = PID(
            float(args.accel_kp),
            float(args.accel_ki),
            float(args.accel_kd),
            float(args.accel_n),
            Ts,
            u_min,
            u_max,
            kb_aw,
            der_on_meas=True,
        )
        accel_pid_high = PID(
            float(args.accel_kp),
            float(args.accel_ki),
            float(args.accel_kd),
            float(args.accel_n),
            Ts,
            u_min,
            u_max,
            kb_aw,
            der_on_meas=True,
        )

        member = PlatoonMember(
            name=name,
            platoon_index=i,
            speed_pid_low=speed_pid_low,
            speed_pid_mid=speed_pid_mid,
            speed_pid_high=speed_pid_high,
            accel_pid_low=accel_pid_low,
            accel_pid_mid=accel_pid_mid,
            accel_pid_high=accel_pid_high,
            desired_time_headway=float(bridge._desired_time_headways.get(name, 0.3)),
            min_spacing=float(bridge._min_spacings.get(name, 5.0)),
            K_dist=float(args.k_dist),
            K_vel=float(args.k_vel),
            K_dist_panic=(None if args.k_dist_panic is None else float(args.k_dist_panic)),
            K_vel_panic=(None if args.k_vel_panic is None else float(args.k_vel_panic)),
            ttc_panic_s=float(args.ttc_panic_s),
            dist_err_deadband_m=float(args.dist_err_deadband_m),
            vel_diff_deadband_mps=float(args.vel_diff_deadband_mps),
            control_period=Ts,
        )

        try:
            member._timer.cancel()
        except Exception:
            pass

        nodes.append(member)

    executor = MultiThreadedExecutor()
    for n in nodes:
        executor.add_node(n)

    def drain_executor(max_callbacks: int, max_wall_s: float) -> None:
        start = time.perf_counter()
        for _ in range(int(max_callbacks)):
            executor.spin_once(timeout_sec=0.0)
            if (time.perf_counter() - start) >= max_wall_s:
                break

    dt_wall_s = float(bridge.dt) if args.wall_dt is None else float(args.wall_dt)
    state_delivery_budget_s = min(0.002, dt_wall_s * 0.25)

    bridge.apply_controls()
    tick_due_wall_s = time.perf_counter()
    ego_name = bridge.ros_names[bridge.ego_index]
    next_dbg_sim_s = (
        float(float(args.print_every_s))
        if float(args.print_every_s) > 0.0
        else float("inf")
    )

    print(
        "Running accelPID platooning with fixed gains:\n"
        f"  speed_pid (kp,ki,kd,N)=({args.speed_kp},{args.speed_ki},{args.speed_kd},{args.speed_n})\n"
        f"  accel_pid (kp,ki,kd,N)=({args.accel_kp},{args.accel_ki},{args.accel_kd},{args.accel_n})\n"
        f"  print_every_s={args.print_every_s} max_sim_time_s={args.max_sim_time_s}"
    )
    try:
        while not bridge.finished:
            while True:
                now_s = time.perf_counter()
                if now_s >= tick_due_wall_s:
                    break
                drain_executor(max_callbacks=200, max_wall_s=state_delivery_budget_s)
                time.sleep(min(0.0005, max(0.0, tick_due_wall_s - now_s)))

            bridge.step_simulation()
            drain_executor(max_callbacks=5000, max_wall_s=state_delivery_budget_s)

            for n in nodes[1:]:
                try:
                    n.step_once()
                except Exception:
                    pass

            drain_executor(max_callbacks=5000, max_wall_s=state_delivery_budget_s)
            bridge.apply_controls()

            dbg = bridge._last_dbg.get(ego_name, None)

            # Live debug print so it doesn't look "stuck"
            if float(bridge.sim_time) >= float(next_dbg_sim_s):
                cmd = bridge._last_cmd.get(ego_name, {"throttle": 0.0, "brake": 0.0})
                throttle = float(cmd.get("throttle", 0.0))
                brake = float(cmd.get("brake", 0.0))
                speed = float(bridge._last_speeds[bridge.ego_index]) if bridge._last_speeds else 0.0
                accel = (
                    float(bridge._last_accels_long[bridge.ego_index]) if bridge._last_accels_long else 0.0
                )
                sp = float(bridge.lead_speed_setpoints[bridge.lead_sp_idx])
                v_err = float(dbg.get("v_err", sp - speed)) if dbg is not None else float(sp - speed)
                a_des = float(dbg.get("a_des", 0.0)) if dbg is not None else 0.0
                a_err = float(dbg.get("a_err", 0.0)) if dbg is not None else 0.0
                print(
                    f"t={bridge.sim_time:7.2f}s sp={sp:5.2f} v={speed:5.2f} v_err={v_err:6.3f} "
                    f"a_des={a_des:6.3f} a={accel:6.3f} a_err={a_err:6.3f} "
                    f"thr={throttle:5.3f} brk={brake:5.3f}"
                )
                next_dbg_sim_s = float(bridge.sim_time) + float(args.print_every_s)

            if float(args.max_sim_time_s) > 0.0 and float(bridge.sim_time) >= float(args.max_sim_time_s):
                bridge.finished = True

            tick_due_wall_s += dt_wall_s
    except KeyboardInterrupt:
        pass
    finally:
        bridge.shutdown()
        for n in nodes:
            try:
                executor.remove_node(n)
            except Exception:
                pass
            try:
                n.destroy_node()
            except Exception:
                pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
