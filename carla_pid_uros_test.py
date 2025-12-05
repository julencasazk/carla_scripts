

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
from PID import PID
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from pid_message.msg import PID
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
        self._pid_param_pub = self.create_publisher(PID, 'pid_params', qos_params)
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
        m = PID()
        m.kp = self._args.kp
        m.ki = self._args.ki
        m.kd = self._args.kd
        m.n  = self._args.N
        self._pid_param_pub.publish(m)
        self.get_logger().info(f"PID params published: {m}")

    def u_cb(self, msg: Float64):
        self._u_val = float(msg.data)

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
    print("Controller setup") # Might need to check if params are propely published

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
    settings.fixed_delta_seconds = 0.01 # I don't know if it's important right now, 
                                        # but should match the sampling
                                        # rate of the PID on the STM32
    settings.no_rendering_mode = False  # Camera feed is prefered, although not needed
    world.apply_settings(settings)
    print(f"Synchronous mode ON, dt = {0.01} s")

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

    # Step test definition
    dt = settings.fixed_delta_seconds
    total_time = 3600.0 # Very long time, enough to input all throttle setpoints 
    total_steps = int(total_time / dt)
    print(f"Running step test: dt={dt}s, total_time={total_time}s")
    print(f"Logging to: {csv_filename}")

    teleport_time = int(5.0 / dt)  # interval in steps
    # For doing controlled tests (no random throttle)
    # Incremental tests: sp1 -> 0.0 -> sp2 -> 0.0 ...
    # setpoints = [0.0, 5.0, 0.0, 10.0, 0.0, 15.0, 0.0, 20.0, 0.0, 25.0, 0.0, 30.0, 0.0]
    # More random values
    setpoints = [0.0, 33.00, 30.00, 24.00, 15.00, 17.00, 22.00, 4.00, 13.34, 0.00, 1.00, 2.00, 13.00, 0.00]

    y = np.zeros_like(total_steps)
    u = np.zeros_like(total_steps)

    idx = 0
    finished = False
    
    try:
        # Sim time counter 
        sim_time = 0.0
        start_timestamp = None
        step = 0
        # Wait the car to stabilize before doing a setpoint
        stable_step = 0        
        prev_speed = 0.0

        last_change_step = 0
        last_change_speed = 0.0
        throttle_cmd = 0.0
        band = 0.2                      # stable band value of +- <band> m/s
                                        # speed is stable if inside band for <min_wait_steps> steps 
        min_wait_steps = int(5.0 / dt)  # secons between stability checks 

        # Vehicle reset 
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # Synchronous loop 
        #while sim_time < total_time:
        while not finished:
            
            # Update a single simulation step 
            world.tick()
            snapshot = world.get_snapshot()
            t = snapshot.timestamp.elapsed_seconds
            if start_timestamp is None:
                start_timestamp = t
            sim_time = t - start_timestamp
            
            # If on last step, do the step and finish
            if sim_time >= total_time:
                finished = True

            brake_cmd = 0.0  

            vel = vehicle.get_velocity()
            accel = vehicle.get_acceleration()
            # Teleport every 5 seconds to stay in flat segment 
            # of highway continuously
            if step > 0 and step % teleport_time == 0:
                vehicle.set_transform(spawn_point)
                print("Teleport!!")
            
            speed = math.sqrt(vel.x**2 + vel.y**2)

            # Publish new PV
            throttle_cmd = node._u_val
            node._y_val = speed 
            pv_msg = Float64()
            pv_msg.data = float(speed)
            node._pv_pub.publish(pv_msg)

            # Publish setpoints
            node._r_val = setpoints[idx]
            r_msg = Float64()
            r_msg.data = float(setpoints[idx])
            node._r_pub.publish(r_msg)
        
            
            # Check if enough time has passed since last change
            if (step - last_change_step) >= min_wait_steps:
                # Has it entered the error band? 
                if abs(speed - setpoints[idx]) <= band:
                    # throttle_cmd = float(np.random.rand())

                    if idx == (len(setpoints) - 1):
                        finished = True
                    idx = (idx+1) % len(setpoints)
                    print(f"Setpoint change to {setpoints[idx]:.3f} at t={sim_time:.2f}s "
                          f"(speed: d{speed:.2f} m/s)")
                last_change_step = step
                last_change_speed = speed

            print(f"Current throttle: {throttle_cmd:.3f}")
            print(f"Current Setpoint: {setpoints[idx]:.3f}")
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
                f"{setpoints[idx]:.4f}",
                f"{brake_cmd:.4f}",
                f"{speed:.4f}",
                f"{accel.x:.4f}",
                f"{accel.y:.4f}",
                f"{accel.z:.4f}",
            ])
            csv_file.flush()
            print(f"Step {step}/{total_steps}")
            print(f"Elapsed time: {sim_time}/{total_time}")

            step = step + 1
            

        print("Step test finished successfully.")

        # Stop vehicle when test over
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        time.sleep(1.0)

    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopping test early.")

    finally:
        print("Cleaning up actors...")
        try:
            cam.stop()
        except:
            pass

        try:
            vehicle.destroy()
        except:
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
