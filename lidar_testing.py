import carla
import numpy as np
import argparse
import open3d as o3d
import time

def main():
    parser = argparse.ArgumentParser(description="Testing LiDAR sensors in CARLA sim with ROS and micro-ROS with a ESP32\nFirst execute another script that spawns a vehicle")
    parser.add_argument('--host', type=str, default='localhost', help="CARLA host IP")
    parser.add_argument('--port', type=int, default=2000, help="CARLA port")
    parser.add_argument('--vehicle', type=str, default='vehicle.tesla.model3', help="Vehicle model from CARLA blueprint library")
    parser.add_argument('--no-vis', action='store_true', help='Disable Open3D visualization (headless mode)')
    args = parser.parse_args()

    # Visualization (optional / headless fallback)
    use_vis = not args.no_vis
    pcd_o3d, vis = None, None
    if use_vis:
        try:
            pcd_o3d = o3d.geometry.PointCloud()
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="LiDAR Point Cloud")
            vis.add_geometry(pcd_o3d)
            vc = vis.get_view_control()
            if vc is not None:
                vc.set_constant_z_far(1000)
            else:
                print("Open3D view control unavailable; disabling visualization.")
                use_vis = False
        except Exception as e:
            print("Open3D visualization unavailable; running headless. Reason:", e)
            use_vis, pcd_o3d, vis = False, None, None

    # CARLA client
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_library = world.get_blueprint_library()

    # Vehicle selection
    vehicle = None
    try:
        matches = world.get_actors().filter(args.vehicle)
        if len(matches) > 0:
            vehicle = matches[0]
        else:
            any_vehicle = world.get_actors().filter('vehicle.*')
            if len(any_vehicle) > 0:
                vehicle = any_vehicle[0]
    except Exception:
        pass
    if vehicle is None:
        raise RuntimeError(f"No actor found with blueprint {args.vehicle}")

    # Autopilot
    try:
        vehicle.set_autopilot(True)
    except Exception:
        pass

    # LiDAR setup
    lidar_bp = bp_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('points_per_second', '56000')
    sensor = None
    try:
        sensor = world.spawn_actor(
            lidar_bp,
            carla.Transform(carla.Location(x=0.0, y=0.0, z=1.0)),
            attach_to=vehicle
        )
    except Exception as e:
        raise RuntimeError(f"Failed to spawn LiDAR: {e}")

    # Callback
    def lidar_cb(point_cloud):
        try:
            data = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
            if data.size == 0:
                print("Received empty point cloud")
                return
            points = data.reshape((-1, 4))[:, :3]
            if use_vis and pcd_o3d is not None and vis is not None:
                pcd_o3d.points = o3d.utility.Vector3dVector(points)
                vis.update_geometry(pcd_o3d)
            print(f"Received {points.shape[0]} points from LiDAR")
        except Exception as e:
            print("LiDAR callback error:", e)

    sensor.listen(lidar_cb)

    # Main loop
    try:
        while True:
            if use_vis and vis is not None:
                vis.poll_events()
                vis.update_renderer()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if sensor is not None:
                sensor.stop()
        except Exception:
            pass
        try:
            if sensor is not None:
                sensor.destroy()
        except Exception:
            pass
        if use_vis and vis is not None:
            try:
                vis.destroy_window()
            except Exception:
                pass

if __name__ == "__main__":
    main()
