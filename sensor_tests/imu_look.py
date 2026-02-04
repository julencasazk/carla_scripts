#! /usr/bin/env python3
"""
IMU visualization tool that renders orientation from ROS IMU messages.

Outputs
  - On-screen orientation view (OpenCV/DPG) and console diagnostics.

Run (example)
  python3 sensor_tests/imu_look.py
"""

import rclpy
import carla
import numpy as np
import cv2
import argparse
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32
from std_msgs.msg import Bool
from sensor_msgs.msg import Imu
import queue
import multiprocessing
import time
import threading
import math
import dearpygui.dearpygui as dpg


use_dpg = False

# Axis Map for quickly changing axis from real IMU to CARLA.
# e.g. rolling the imu left pitches carla up: 'PITCH_FROM': ('roll', -1)
AXIS_MAP = {
    'ROLL_FROM':  ('pitch', +1),
    'PITCH_FROM': ('roll', +1), 
    'YAW_FROM':   ('yaw',  -1), 
}


# --- 3D orientation helpers (wireframe cube + axes projected to 2D) ---

def _rpy_deg_to_rotmat(roll_deg: float, pitch_deg: float, yaw_deg: float):
    rx = math.radians(roll_deg)
    ry = math.radians(pitch_deg)
    rz = math.radians(yaw_deg)

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx,  cx]], dtype=np.float32)
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)

def _project_points(points_xyz: np.ndarray, width: int, height: int, f: float = 180.0, cam_d: float = 4.0):
    # Perspective projection onto drawlist space
    cx, cy = width * 0.5, height * 0.5
    xs, ys = [], []
    for x, y, z in points_xyz:
        zc = z + cam_d
        zc = 0.01 if zc <= 0.01 else zc
        s = f / zc
        xs.append(cx + x * s)
        ys.append(cy - y * s)
    return np.stack([xs, ys], axis=1)

def _draw_orientation(dl_tag: int, roll: float, pitch: float, yaw: float, size=(320, 320)):
    w, h = size
    dpg.delete_item(dl_tag, children_only=True)

    # Model: unit cube and axes
    cube = np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
    ], dtype=np.float32) * 0.9
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom
        (4,5),(5,6),(6,7),(7,4),  # top
        (0,4),(1,5),(2,6),(3,7),  # verticals
    ]
    origin = np.array([[0,0,0]], dtype=np.float32)
    axes = np.array([
        [1.5, 0.0, 0.0],  # X red
        [0.0, 1.5, 0.0],  # Y green
        [0.0, 0.0, 1.5],  # Z blue
    ], dtype=np.float32)

    # Rotate
    R = _rpy_deg_to_rotmat(roll, pitch, yaw)
    cube_r = (cube @ R.T)
    origin_r = (origin @ R.T)
    axes_r = (axes @ R.T)

    # Project
    cube_2d = _project_points(cube_r, w, h)
    origin_2d = _project_points(origin_r, w, h)[0]
    axes_2d = _project_points(axes_r, w, h)

    # Draw cube
    for a, b in edges:
        dpg.draw_line(cube_2d[a], cube_2d[b], color=(200, 200, 200, 255), thickness=2, parent=dl_tag)

    # Draw axes from origin
    colors = [(255, 80, 80, 255), (80, 255, 80, 255), (80, 160, 255, 255)]  # X, Y, Z
    for i in range(3):
        dpg.draw_line(origin_2d, axes_2d[i], color=colors[i], thickness=3, parent=dl_tag)

    # Axis labels
    label_offset = 6
    labels = ["X", "Y", "Z"]
    for i in range(3):
        p = axes_2d[i]
        dpg.draw_text((p[0] + label_offset, p[1] + label_offset), labels[i], color=colors[i], size=14, parent=dl_tag)

def dash_cam(args, img_queue, img_lock, imu_data_queue):
    if args.dpg:
        dpg.create_context()
        try:
            tex_tag = None
            img_item = None
            cur_w = cur_h = 0
            last_rpy = (0.0, 0.0, 0.0)  # roll, pitch, yaw (deg)

            # Wait for first frame (blocking, no backlog created)
            first_payload = None
            while first_payload is None:
                try:
                    first_payload = img_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if first_payload is None:
                    dpg.destroy_context()
                    return

            raw_bytes, width, height = first_payload
            arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width, 4))
            rgba = arr[:, :, [2, 1, 0, 3]].astype(np.float32) / 255.0
            init_data = rgba.flatten().tolist()
            cur_w, cur_h = width, height

            with dpg.texture_registry():
                tex_tag = dpg.add_dynamic_texture(cur_w, cur_h, init_data)

            with dpg.window(label="Dashboard Camera", tag="cam_window"):
                img_item = dpg.add_image(tex_tag, width=cur_w, height=cur_h)

            # IMU text window
            max_angular = [0,0,0]
            max_linear = [0,0,0]
            max_orientation = [0,0,0,0]
            with dpg.window(label="IMU_Data", tag="data_window", pos=(cur_w + 8, 0), width=340, height=200):
                text_angular_vel = dpg.add_text("Angular Velocity: (0, 0, 0) [x, y, z]")
                text_max_angular_vel = dpg.add_text("MAX Angular: (0, 0, 0) [x, y, z]")
                text_linear_acc = dpg.add_text("Linear Acceleration: (0, 0, 0) [x, y, z]")
                text_max_linear_acc = dpg.add_text("MAX Linear: (0, 0, 0) [x, y, z]")
                text_orientation = dpg.add_text("Orientation: (0, 0, 0, 0) [x, y, z, w]")
                text_max_orientation = dpg.add_text("MAX Orientation: (0, 0, 0, 0) [x, y, z, w]")
                reset_button = dpg.add_button(label="Reset MAX values", width=150)

            # Orientation wireframe window + drawlist
            with dpg.window(label="IMU Orientation", tag="orient_window", pos=(cur_w + 8, 210), width=340, height=360):
                orient_drawlist = dpg.add_drawlist(width=320, height=320, tag="orient_drawlist")

            dpg.create_viewport(
                title="Dashboard Camera",
                width=cur_w + 16 + 360,
                height=max(cur_h + 39, 210 + 360)
            )
            dpg.setup_dearpygui()
            dpg.show_viewport()

            while dpg.is_dearpygui_running():
                # Drain to latest frame
                payload = None
                try:
                    while True:
                        payload = img_queue.get_nowait()
                except queue.Empty:
                    pass

                if payload:
                    raw_bytes, width, height = payload
                    if width != cur_w or height != cur_h:
                        if tex_tag is not None:
                            dpg.delete_item(tex_tag)
                        with dpg.texture_registry():
                            tex_tag = dpg.add_dynamic_texture(width, height, [0.0] * (width * height * 4))
                        dpg.configure_item(img_item, texture_tag=tex_tag, width=width, height=height)
                        cur_w, cur_h = width, height

                    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width, 4))
                    rgba = arr[:, :, [2, 1, 0, 3]].astype(np.float32) / 255.0
                    dpg.set_value(tex_tag, rgba.flatten().tolist())

                # Drain to latest IMU payload
                data_payload = None
                try:
                    while True:
                        data_payload = imu_data_queue.get_nowait()
                except queue.Empty:
                    pass

                if data_payload:
                    imu_msg, roll_carla, pitch_carla, yaw_carla = data_payload
                    last_rpy = (roll_carla, pitch_carla, yaw_carla)
                    
                    if abs(imu_msg.angular_velocity.x) > max_angular[0]:
                        max_angular[0] = abs(imu_msg.angular_velocity.x)
                    if abs(imu_msg.angular_velocity.y) > max_angular[1]:
                        max_angular[1] = abs(imu_msg.angular_velocity.y)
                    if abs(imu_msg.angular_velocity.z) > max_angular[2]:
                        max_angular[2] = abs(imu_msg.angular_velocity.z)
                    
                    if abs(imu_msg.linear_acceleration.x) > max_linear[0]:
                        max_linear[0] = abs(imu_msg.linear_acceleration.x)
                    if abs(imu_msg.linear_acceleration.y) > max_linear[1]:
                        max_linear[1] = abs(imu_msg.linear_acceleration.y)
                    if abs(imu_msg.linear_acceleration.z) > max_linear[2]:
                        max_linear[2] = abs(imu_msg.linear_acceleration.z)
                    
                    if abs(imu_msg.orientation.x) > max_orientation[0]:
                        max_orientation[0] = abs(imu_msg.orientation.x)
                    if abs(imu_msg.orientation.y) > max_orientation[1]:
                        max_orientation[1] = abs(imu_msg.orientation.y)
                    if abs(imu_msg.orientation.z) > max_orientation[2]:
                        max_orientation[2] = abs(imu_msg.orientation.z)
                    if abs(imu_msg.orientation.w) > max_orientation[3]:
                        max_orientation[3] = abs(imu_msg.orientation.w)
                        
                    # Button handling
                    if dpg.is_item_clicked(reset_button):
                        max_angular = [0,0,0]
                        max_linear = [0,0,0]
                        max_orientation = [0,0,0,0]
                    dpg.set_value(text_max_angular_vel,
                                  f"MAX Angular: ({max_angular[0]:.2f}, {max_angular[1]:.2f}, {max_angular[2]:.2f}) [x, y, z]")
                    dpg.set_value(text_max_linear_acc,
                                  f"MAX Linear: ({max_linear[0]:.2f}, {max_linear[1]:.2f}, {max_linear[2]:.2f}) [x, y, z]")
                    dpg.set_value(text_max_orientation,
                                  f"MAX Orientation: ({max_orientation[0]:.2f}, {max_orientation[1]:.2f}, {max_orientation[2]:.2f}, {max_orientation[3]:.2f}) [x, y, z, w]")

                    dpg.set_value(text_angular_vel,
                                  f"Angular Velocity: ({imu_msg.angular_velocity.x:.2f}, {imu_msg.angular_velocity.y:.2f}, {imu_msg.angular_velocity.z:.2f}) [x, y, z]")
                    dpg.set_value(text_linear_acc,
                                  f"Linear Acceleration: ({imu_msg.linear_acceleration.x:.2f}, {imu_msg.linear_acceleration.y:.2f}, {imu_msg.linear_acceleration.z:.2f}) [x, y, z]")
                    dpg.set_value(text_orientation,
                                  f"Orientation: ({imu_msg.orientation.x:.2f}, {imu_msg.orientation.y:.2f}, {imu_msg.orientation.z:.2f}, {imu_msg.orientation.w:.2f}) [x, y, z, w]")

                # Update orientation drawlist every frame using last_rpy
                _draw_orientation(orient_drawlist, *last_rpy, size=(320, 320))

                dpg.render_dearpygui_frame()
                time.sleep(0.001)

        finally:
            dpg.destroy_context()
    else:
        # Dashcam using cv2
        cv2.namedWindow("Dashboard Camera", cv2.WINDOW_NORMAL)
        try:
            while True:
                try:
                    payload = img_queue.get_nowait()
                except queue.Empty:
                    time.sleep(0.01)
                    continue
                if payload is None:
                    break

                raw_bytes, width, height = payload
                array = np.frombuffer(raw_bytes, dtype=np.uint8)
                array = np.reshape(array, (height, width, 4))
                array = array[:, :, :3]  # Discard alpha

                cv2.imshow("Dashboard Camera", array)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            print("Cleaning up OpenCV...")
            cv2.destroyAllWindows()
            print("Destroyed all windows")

def quat_to_euler_deg(x, y, z, w):
    # Normalize quaternion
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n == 0.0:
        return 0.0, 0.0, 0.0
    x, y, z, w = x/n, y/n, z/n, w/n

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

class CarlaRosNode(Node):
    def __init__(self, camera, base_transform, abs_rot_enable, imu_data_queue):
        super().__init__('carla_cam_control')
        qos = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT)
        self._imu_sub = self.create_subscription(Imu, 'imu_data', self.imu_cb, qos)
        self._camera = camera
        self._base_loc = base_transform.location
        self._base_rot = base_transform.rotation
        self._last_r_m = None 
        self._last_p_m = None 
        self._last_y_m = None
        self._abs_rot_enable = abs_rot_enable
        self._first_reading_done = False
        self._imu_data_queue = imu_data_queue
        
    # Replace axis between them following the map
    def _apply_axis_map(self, roll, pitch, yaw):
        vals = {'roll': roll, 'pitch': pitch, 'yaw': yaw}
        def pick(which, sign):
            return sign * vals[which]
        r = pick(*AXIS_MAP['ROLL_FROM'])
        p = pick(*AXIS_MAP['PITCH_FROM'])
        y = pick(*AXIS_MAP['YAW_FROM'])
        return r, p, y 


    def imu_cb(self, msg: Imu):
        # IMU quaternion -> Euler (deg)
        ox, oy, oz, ow = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        r, p, y = quat_to_euler_deg(ox, oy, oz, ow)
        
        r_m, p_m, y_m = self._apply_axis_map(r, p, y)
        
        if (not self._first_reading_done):
            self._last_p_m = p_m
            self._last_r_m = r_m
            self._last_y_m = y_m
            self._first_reading_done = True

        if not self._abs_rot_enable:
            r_m = r_m - self._last_r_m
            p_m = p_m - self._last_p_m 
            y_m = y_m - self._last_y_m

        roll_carla  = self._base_rot.roll  + r_m
        pitch_carla = self._base_rot.pitch + p_m
        yaw_carla   = self._base_rot.yaw   + y_m
        

        r_msg, p_msg, y_msg  = pitch_carla, yaw_carla, roll_carla
        imu_data = [
            msg,
            r_msg,
            p_msg,
            y_msg
        ]
        
        try:
            while True:
                self._imu_data_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._imu_data_queue.put_nowait(imu_data)
        except queue.Full:
            pass

        new_rot = carla.Rotation(pitch=pitch_carla, yaw=yaw_carla, roll=roll_carla)
        self._camera.set_transform(carla.Transform(self._base_loc, new_rot))
        
    
        
    
def ros_background_spin(camera, base_transform, abs_rot_enable, imu_data_queue):
    rclpy.init()
    node = CarlaRosNode(camera, base_transform, abs_rot_enable, imu_data_queue)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    t = threading.Thread(target=executor.spin, daemon=True)
    t.start()
    return executor, node, t
        
        

def main(args, img_queue, img_lock, imu_data_queue):

    print("CARLA Camera controller with IMU")
    print("Mash Ctrl+C to exit, idk")
    
    # CARLA Client setup
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_library = world.get_blueprint_library()
    
    vehicle_bp = bp_library.find('vehicle.tesla.model3')
    vehicle = world.spawn_actor(vehicle_bp,
                                np.random.choice(world.get_map().get_spawn_points()))
    # ================
    # Comment for manual control

    #tm = client.get_trafficmanager()
    vehicle.set_autopilot(True)

    # =================
    print(f"Spawned {vehicle.type_id}")

    # Attach a camera RGB sensor to the side of the car
    cam_bp = bp_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '480')
    cam_transform = carla.Transform(carla.Location(x=0.8, y=1.2, z=1.0), carla.Rotation(yaw=15))
    cam = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    print(f"Spawned {cam.type_id}")
    
    # When camera gets an image, pass it to CV2 process
    def cam_callback(img):
        frame = (bytes(img.raw_data), img.width, img.height)
        try:
            while True:
                img_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            img_queue.put_nowait(frame)
        except queue.Full:
            # Rare race; drop and move on
            pass

    cam.listen(cam_callback)

    executor, node, ros_t = ros_background_spin(cam, cam_transform, args.abs, imu_data_queue)
        
    print(f"Waiting for sensors to start...")
    time.sleep(5)
    print(f"Reading ") 
    
    try:
        # Main blocking loop
        while rclpy.ok():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            print("Trying to exit in a controlled way")
            cam.stop()
            cam.destroy()
            vehicle.destroy()
            executor.shutdown(timeout_sec=1.0)
            node.destroy_node()
            rclpy.shutdown()
            # Send shutdown sentinel without blocking
            try:
                while True:
                    img_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                img_queue.put_nowait(None)
            except queue.Full:
                pass
        except Exception:
            print(f"Error exiting correctly!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Publish vehicle data and control the vehicle with ROS")
    parser.add_argument('--port', type=int, default=2000, help="CARLA port")
    parser.add_argument('--host', type=str, default='localhost', help="CARLA port")
    parser.add_argument('--abs', action='store_true', help="Use absolute rotation from IMU")
    parser.add_argument('--dpg', action='store_true', help="Use DearPyGUI for dashcam instead of OpenCV")
    args = parser.parse_args()

    # Only keep the latest frame
    img_queue = multiprocessing.Queue(maxsize=1)
    img_lock = multiprocessing.Lock()  # unused
    
    imu_data_queue = multiprocessing.Queue(maxsize=1)
    
    processes = [
        multiprocessing.Process(target=main, args=(args, img_queue, img_lock, imu_data_queue)),
        multiprocessing.Process(target=dash_cam, args=(args, img_queue, img_lock, imu_data_queue))
    ]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()
