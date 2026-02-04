# __Python scripts for CARLA Simulator__

## Requirements

The scripts are tested on:

- Ubuntu 22.04 through WSL2
- Python 3.10
- CARLA 0.9.16 prebuilt binaries
- CARLA 0.9.16 client libraries for Python 3.10

## Usage
This project has been used with a Python venv environment. Set it up with and source it with the provided `requirements.txt`.
```
python -m venv .carlaenv
source .carlaenv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
core/                    # Main platoon controller + primary CARLA/ROS tests
lib/                     # Shared modules used across scripts
tuning/                  # Data collection + PSO tuning utilities
  carla_step_tests/      # CARLA step tests for system identification
  pso/                   # Offline PSO tuning + debug tools
cascade_PID_NOT_WORKING/ # Cascaded PID experiments (archived)
experiments/             # One-off experiments and prototypes
tools/                   # Helper utilities (spawnpoints, logging, plotting, etc.)
sensor_tests/            # IMU/LiDAR/camera tests
```
Each file starts with a comment block describing the function of said file.

## Main files to test

The latest version of the working system utilizes the following files:
```
core/following_ros_cam.py
tools/plot_csv.py
```
