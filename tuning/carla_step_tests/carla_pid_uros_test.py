"""
CARLA test with ROS2 topics for PID input/output (micro-ROS style).

Outputs
  - Console logs and optional CSV.

Run (example)
  python3 tuning/carla_step_tests/carla_pid_uros_test.py
"""

import carla
import numpy as np
import cv2
import argparse
import queue
import multiprocessing
import time
import threading
import math
import os
import csv
import control as ctl
from lib.PID import PID
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from pid_message.msg import PID as PIDMsg
from threading import Thread

class CarlaROSPIDNode(Node):
    def __init__(self, args):
        super().__init__('carla_node')
        self.get_logger().info("Carla ROS Node started (real-time mode).")
        self._args = args
        # QoS profiles
        qos_params = rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE)
        qos_stream = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT)
        # Publishers
        self._pv_pub  = self.create_publisher(Float64, 'pv',  qos_stream)
        self._pid_param_pub = self.create_publisher(PIDMsg, 'pid_params', qos_params)
        self._r_pub = self.create_publisher(Float64, 'r', qos_params)

        # Subscribers
        self._u_sub = self.create_subscription(Float64, 'u', self.u_cb, qos_stream)
        # Internal signals
        self._r_val = 0.0
        self._u_val = 0.0
        self._y_val = 0.0
        # Timing
        self.Ts = args.Ts
        self.publish_params()

    def publish_params(self):
        m = PIDMsg()
        m.kp = self._args.kp
        m.ki = self._args.ki
        m.kd = self._args.kd
        m.n  = self._args.N
        self._pid_param_pub.publish(m)
        self.get_logger().info(f"PID params published: {m}")

    def u_cb(self, msg: Float64):
        self._u_val = float(msg.data)


#####################################
# Smooth setpoint generator
#####################################
def smooth_setpoint(t: float) -> float:
    """
    Smooth longitudinal speed setpoint [m/s] as a function of sim time t [s].

    Example profile:
      - 0 -> 15 m/s, transition centered at 10 s
      - 15 -> 25 m/s, transition centered at 30 s
      - 25 ->  5 m/s, transition centered at 50 s

    All transitions are sigmoids, so r(t) is smooth (C^1).
    Tune targets / centers / widths as desired.
    """
    # Start at 0 m/s
    r = 0.0

    # (target_speed_m_s, center_time_s, width_s)
    transitions = [
        (5.0, 10.0,  3.0),
    ]

    for target, center, width in transitions:
        # logistic sigmoid
        alpha = 1.0 / (1.0 + math.exp(-(t - center) / width))
        # blend previous value towards new target
        r = r * (1.0 - alpha) + target * alpha

    return r


#####################################
# Dashcam process 
#####################################
def dashcam(args, img_queue, char_queue, img_lock):
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


#####################################
# Main CARLA process 
#####################################
def main(args, img_queue, char_queue, img_lock):
    print("CARLA longitudinal test with synchronous mode")
    print("Use CTRL+C to exit")

    print("Setting up PID controller...")

    rclpy.init()
    node = CarlaROSPIDNode(args)

    # Run ROS executor in background thread
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    exec_thread = Thread(target=executor.spin, daemon=True)
    exec_thread.start() 
    print("Controller setup")

    latest_frame = None
    frame_lock = threading.Lock()

    # CARLA Client setup
    print("Setting up CARLA Client...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    print("Available maps:")
    for m in client.get_available_maps():
        print("  ", m)

    # Town04 has the longest highways strips of any default map 
    client.load_world('Town04')
    time.sleep(2.0)
    world = client.get_world()
    bp_library = world.get_blueprint_library()

    # Enable synchronous mode with dt=0.01s
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True  
    settings.fixed_delta_seconds = 0.01
    settings.no_rendering_mode = False  # camera preferred, but not required
    world.apply_settings(settings)
    print(f"Synchronous mode ON, dt = {settings.fixed_delta_seconds} s")

    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[54]
    spawn_point.location.z -= 0.3
    vehicle_bp = bp_library.find('vehicle.tesla.model3')
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"Spawned {vehicle.type_id} at spawn point {spawn_point}")
    vehicle.set_autopilot(False)

    cam_bp = bp_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '480')
    cam_bp.set_attribute('fov', '90')

    cam_transform = carla.Transform(carla.Location(x=1.2, y=0.0, z=1.2))
    cam = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    print(f"Spawned {cam.type_id}")

    os.makedirs("spawn_point_imgs", exist_ok=True)

    def cam_cb(img):
        frame = (bytes(img.raw_data), img.width, img.height, img.frame)
        with frame_lock:
            nonlocal latest_frame
            latest_frame = frame

        try:
            while True:
                img_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            img_queue.put_nowait(frame)
        except queue.Full:
            pass

    cam.listen(cam_cb)

    # CSV File for logging vehicle speed to throttle
    csv_filename = args.f 
    csv_file = open(csv_filename, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    # Header 
    csv_writer.writerow([
        "time_s",
        "throttle",
        "setpoint",
        "brake",
        "speed_m_s",
        "accel_x_m_s2",
        "accel_y_m_s2",
        "accel_z_m_s2"
    ])
    csv_file.flush()

    # Test definition
    dt = settings.fixed_delta_seconds
    total_time = 3600.0  # long enough
    total_steps = int(total_time / dt)
    print(f"Running smooth setpoint test: dt={dt}s, total_time={total_time}s")
    print(f"Logging to: {csv_filename}")

    teleport_time = int(5.0 / dt)  # interval in steps

    finished = False
    
    try:
        # Sim time counter 
        sim_time = 0.0
        start_timestamp = None
        step = 0

        # Vehicle reset 
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # Synchronous loop 
        while not finished:
            # Update a single simulation step 
            world.tick()
            snapshot = world.get_snapshot()
            t = snapshot.timestamp.elapsed_seconds
            if start_timestamp is None:
                start_timestamp = t
            sim_time = t - start_timestamp
            
            # Finish after total_time
            if sim_time >= total_time:
                finished = True

            brake_cmd = 0.0  

            vel = vehicle.get_velocity()
            accel = vehicle.get_acceleration()

            # Teleport every 5 seconds to stay in a flat segment of highway
            if step > 0 and step % teleport_time == 0:
                vehicle.set_transform(spawn_point)
                print("Teleport!!")
            
            speed = math.sqrt(vel.x**2 + vel.y**2)

            # PV from CARLA -> ROS
            throttle_cmd = node._u_val
            node._y_val = speed 
            pv_msg = Float64()
            pv_msg.data = float(speed)
            node._pv_pub.publish(pv_msg)

            # Smooth setpoint as function of sim_time
            sp = smooth_setpoint(sim_time)

            # Publish setpoint to ROS
            node._r_val = sp
            r_msg = Float64()
            r_msg.data = float(sp)
            node._r_pub.publish(r_msg)

            print(f"Current throttle: {throttle_cmd:.3f}")
            print(f"Current Setpoint: {sp:.3f}")
            print(f"Current speed: {speed:.3f}")

            control = carla.VehicleControl(
                throttle=float(throttle_cmd),
                brake=float(brake_cmd),
                steer=0.0,
                hand_brake=False,
                reverse=False
            )
            vehicle.apply_control(control)

            csv_writer.writerow([
                f"{sim_time:.4f}",
                f"{throttle_cmd:.4f}",
                f"{sp:.4f}",
                f"{brake_cmd:.4f}",
                f"{speed:.4f}",
                f"{accel.x:.4f}",
                f"{accel.y:.4f}",
                f"{accel.z:.4f}",
            ])
            csv_file.flush()
            print(f"Step {step}/{total_steps}")
            print(f"Elapsed time: {sim_time}/{total_time}")

            step += 1

        print("Smooth setpoint test finished successfully.")

        # Stop vehicle when test over
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        time.sleep(1.0)

    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopping test early.")

    finally:
        print("Cleaning up actors...")
        try:
            cam.stop()
        except Exception:
            pass

        try:
            vehicle.destroy()
        except Exception:
            pass

        csv_file.close()

        print("Done cleanup.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA longitudinal identification test")
    parser.add_argument('--host', type=str, help="CARLA host", default='localhost')
    parser.add_argument('--port', type=int, help="CARLA port", default=2000)
    parser.add_argument('-f', type=str, help="Output csv filename", default="out.csv")
    parser.add_argument('--kp', type=float, default=0.47446807, help="PID proportional gain")
    parser.add_argument('--ki', type=float, default=0.15472707, help="PID integral gain")
    parser.add_argument('--kd', type=float, default=7.83353195, help="PID derivative gain")
    parser.add_argument('--N', type=float, default=20.0, help="PID derivative low pass filter time constant. N = 1/tau")
    parser.add_argument('--Ts', type=float, default=0.01, help="Sampling period (s)")
    args = parser.parse_args()

    img_lock = multiprocessing.Lock()
    char_queue = multiprocessing.Queue()
    img_queue = multiprocessing.Queue(maxsize=1)

    processes = [
        multiprocessing.Process(target=main, args=(args, img_queue, char_queue, img_lock)),
        multiprocessing.Process(target=dashcam, args=(args, img_queue, char_queue, img_lock)),
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
