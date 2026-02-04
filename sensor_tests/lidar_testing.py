"""
CARLA LiDAR test with rolling point cloud visualization (Open3D).

Outputs
  - Open3D window with accumulated point cloud.

Run (example)
  python3 sensor_tests/lidar_testing.py --host localhost --port 2000
"""

import carla
import numpy as np
import open3d as o3d
import time
import threading
import argparse
from collections import deque


def main():
    p = argparse.ArgumentParser(description="CARLA LiDAR sensor testing")
    p.add_argument('--host', type=str, default='localhost', help="CARLA Server Host IP")
    p.add_argument('--port', type=int, default=2000, help="CARLA Server Host TCP Port")
    p.add_argument('-n', type=int, default=20, help="Number of LiDAR frames for rolling accumulation")
    args = p.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    vehicle_bp = np.random.choice(bp_lib.filter('vehicle'))
    vehicle = world.spawn_actor(vehicle_bp, np.random.choice(world.get_map().get_spawn_points()))
    vehicle.set_autopilot(True)
    print(f"Spawned {vehicle.type_id}")

    # Using an already spawned vehicle
    vehicles = world.get_actors().filter('vehicle.*model3*')
    if len(vehicles) == 0:
        vehicles = world.get_actors().filter('vehicle.*')
    vehicle = vehicles[0]
    print(f"Using vehicle: {vehicle.type_id}")


    # Values similar to Velodyne VLP-16 "Puck"
    lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '12')
    lidar_bp.set_attribute('rotation_frequency', '5')
    lidar_bp.set_attribute('points_per_second', '8000')
    lidar_bp.set_attribute('channels', '16')
    lidar_bp.set_attribute('upper_fov', '1')
    lidar_bp.set_attribute('lower_fov', '-1')

    lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    print("LiDAR sensor attached.")

    lock = threading.Lock()
    recent_frames = deque(maxlen=args.n)

    def lidar_callback(point_cloud):
        # global recent_frames
        pts = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
        pts = np.reshape(pts, (int(pts.shape[0] / 4), 4))[:, :3]
                    
        data_size = pts.nbytes
        size_kb = data_size / 1024
        print(f"Size of data: {size_kb} KB")

        lidar_world_matrix = np.array(lidar.get_transform().get_matrix())
        pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
        pts_world = (lidar_world_matrix @ pts_h.T).T[:, :3]
        with lock:
            recent_frames.append(pts_world)

    lidar.listen(lidar_callback)

    vis = o3d.visualization.Visualizer()
    vis.create_window("LiDAR Rolling View", width=960, height=720)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0

    view_ctl = vis.get_view_control() # To center the vis on the car

    print("Streaming LiDAR data with rolling accumulation...")

    first_frame = True
    try:
        while True:
            try:
                with lock:
                    if len(recent_frames) == 0:
                        continue
                    # Concatenate last N frames
                    pts_combined = np.vstack(recent_frames)


                car_tf = vehicle.get_transform()
                car_loc = car_tf.location
                car_rot = car_tf.rotation
                car_pos = np.array([car_loc.x, car_loc.y, car_loc.z])
                dists = np.linalg.norm(pts_combined - car_pos, axis=1)

                # Color by height for clarity
                d_min, d_max = np.min(dists), np.max(dists)
                colors = (dists - d_min) / (d_max - d_min + 1e-6)
                colors = np.stack([colors, 1 - colors, np.zeros_like(colors)], axis=1)
                

                pcd.points = o3d.utility.Vector3dVector(pts_combined)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                
                
                distance = 20.0
                height = 10.0
                
                view_ctl.set_lookat(car_pos)
                

                if first_frame:
                    vis.reset_view_point(True)
                    first_frame = False

                time.sleep(0.05)
            except: 
                print("No points available for this frame")
                pass

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        lidar.stop()
        lidar.destroy()
        vis.destroy_window()
        print("LiDAR and viewer closed.")

    
if __name__ == "__main__":
    main()
