import argparse
import math
from multiprocessing import Queue
import multiprocessing
import queue
import threading
import time
from functools import partial

import carla
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32

from PID import PID
from PlatooningROS import PlatoonMember


#####################################
# Dashcam process 
#####################################
def dashcam(img_queue):
    time.sleep(5)
    cv2.namedWindow("Dashboard camera", cv2.WINDOW_NORMAL)
    payload = None
    try:
        while True:
            try:
                payload = img_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if payload is None:
                continue

            try:
                raw_bytes, width, height = payload[:3]
                array = np.frombuffer(raw_bytes, dtype=np.uint8)
                array = np.reshape(array, (height, width, 4))
                array = array[:, :, :3]  # drop alpha

                cv2.imshow("Dashboard camera", array)
                if cv2.waitKey(1) == ord('q'):
                    break
            except KeyboardInterrupt:
                print("Keyboard Interrupt caught!")
                print("Cleaning up OpenCV...")
                cv2.destroyAllWindows()
                print("Destroyed all windows")

    finally:
        print("Exiting dashcam")
        print("Cleaning up OpenCV...")
        cv2.destroyAllWindows()
        print("Destroyed all windows")


class CarlaPlatoonBridge(Node):
    def __init__(
        self,
        host: str,
        port: int,
        plen: int,
        mcu_index: int,
        dt: float = 0.05,
        img_queue=None
    ):
        super().__init__("carla_platoon_bridge")

        if plen < 1:
            raise ValueError("Platoon length must be >= 1")
        if not (0 <= mcu_index < plen):
            raise ValueError("mcu_index must be in [0, plen-1]")

        self._plen = plen
        self._mcu_index = mcu_index
        self._dt = dt
        self._img_queue = img_queue

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

        self._veh_names = []
        self._vehicles = []
        self._speed_pubs = {}
        self._dist_pubs = {}
        self._throttle_subs = {}
        self._brake_subs = {}
        self._last_cmd = {}
        self._camera = None
        self._camera_tf = None

        for i in range(plen):
            if i == 0:
                ros_name = "veh_lead"
            elif i == mcu_index:
                ros_name = "veh_ego"
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

            # Use functools.partial instead of a lambda so each callback
            # is bound cleanly to its vehicle name.
            self._throttle_subs[ros_name] = self.create_subscription(
                Float32,
                f"{ros_name}/command/throttle",
                partial(self._throttle_cb, name=ros_name),
                qos,
            )
            self._brake_subs[ros_name] = self.create_subscription(
                Float32,
                f"{ros_name}/command/brake",
                partial(self._brake_cb, name=ros_name),
                qos,
            )

        if self._img_queue is not None and len(self._vehicles) > 0:
            cam_bp = bp_library.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", "640")
            cam_bp.set_attribute("image_size_y", "480")
            cam_bp.set_attribute("fov", "90")


            cam_transform = carla.Transform(carla.Location(x=0.0, y=10.0, z=10.0), carla.Rotation(yaw=-20.0, pitch=-35))


            last_vehicle = self._vehicles[-1]
            self._camera_tf = cam_transform
            self._camera = world.spawn_actor(
                cam_bp, cam_transform, attach_to=last_vehicle
            )
            self.get_logger().info(
                f"Spawned {self._camera.type_id} attached to {self._veh_names[-1]}"
            )

            self._camera.listen(self._camera_cb)

        self._timer = self.create_timer(self._dt, self._step)

    def _throttle_cb(self, msg: Float32, name: str):
        value = float(msg.data)
        self._last_cmd[name]["throttle"] = value
        # Debug: confirm that throttle commands are received by the bridge
        print(f"[CB throttle] {name} <- {value:.2f}")

    def _brake_cb(self, msg: Float32, name: str):
        value = float(msg.data)
        self._last_cmd[name]["brake"] = value
        # Debug: confirm that brake commands are received by the bridge
        print(f"[CB brake] {name} <- {value:.2f}")

    def _step(self):
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

            print(f"[CARLA {name}] t={t:.2f}s v={speeds[idx]:.2f} m/s ")
            print(f"dist_prev={dists[idx]:.2f} m throttle={throttle:.2f} brake={brake:.2f}")

    def _camera_cb(self, image: carla.Image):
        """Camera callback to push latest frame into multiprocessing queue."""
        if self._img_queue is None:
            return

        frame = (bytes(image.raw_data), image.width, image.height, image.frame)

        try:
            while True:
                self._img_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._img_queue.put_nowait(frame)
        except queue.Full:
            pass

    def destroy(self):
        self.get_logger().info("Destroying CARLA actors...")

        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
            try:
                self._camera.destroy()
            except Exception:
                pass

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


def ros_p(img_queue):
    parser = argparse.ArgumentParser(description="CARLA–ROS platoon bridge with optional MCU")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--plen", type=int, default=2, help="Number of platoon members")
    parser.add_argument(
        "--mcu-index",
        type=int,
        default=1,
        help="Index [0..plen-1] of the MCU (micro-ROS) vehicle in the platoon",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Control / CARLA step [s]")
    args = parser.parse_args()

    rclpy.init()

    bridge = CarlaPlatoonBridge(
        host=args.host,
        port=args.port,
        plen=args.plen,
        mcu_index=args.mcu_index,
        dt=args.dt,
        img_queue=img_queue,
    )

    nodes = [bridge]

    kp = 0.26
    ki = 0.13
    kd = 0.006
    N_d = 5.0
    u_min, u_max = -1.0, 1.0
    kb_aw = 1.0

    for i, name in enumerate(bridge._veh_names):
        if i == args.mcu_index:
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


def main():
    img_queue = Queue(maxsize=1)

    ps = [
        multiprocessing.Process(target=ros_p, args=(img_queue,)),
        multiprocessing.Process(target=dashcam, args=(img_queue,)),
    ]

    for p in ps:
        p.start()
    for p in ps:
        p.join()


if __name__ == "__main__":
    main()
