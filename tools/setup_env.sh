#!/usr/bin/bash
# =======================================
# Script: setup_env.sh
# Author: julencasazk
# Description:
#   A script to automatically setup ROS2 and fix
#   a libstdc++ conflict when launching a specific
#   conda environment
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh <conda_env_name>
# =======================================
set -e

# If missing argument
if [ -z "$1" ]; then
  echo "Usage: $0 <conda_env_name>"
  echo "Example: ./setup_env.sh carla"
  exit 1
fi

CONDA_ENV_NAME="$1"

# Check if conda env exists
echo "Looking for conda env: '$CONDA_ENV_NAME' ..."
CONDA_PREFIX=$(conda env list | grep -E "^$CONDA_ENV_NAME\s" | awk '{print $2}')

if [ -z "$CONDA_PREFIX"]; then
  echo "Could not find environment '$CONDA_ENV_NAME'."
  echo "First create it with:"
  echo ""
  echo "conda create -f environment.yml"
  exit 1
fi

echo "Found encironment with name '$CONDA_ENV_NAME'."

# Create conda activation folders and scripts
echo "Creating conda activation and deactivation hook folders"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"

ACTIVATE_FILE="$CONDA_PREFIX/etc/conda/activate.d/ros_setup.sh"
cat >"$ACTIVATE_FILE" <<'EOF'
#!/usr/bin/env bash

if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
else
    echo "Could not find ROS2 setup script in /opt/ros/humble/setup.bash"
fi

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

echo "Successfuly integrated ROS2 and conda env"
EOF
chmod +x "$ACTIVATE_FILE"

DEACTIVATION_FILE="$CONDA_PREFIX/etc/conda/deactivate.d/cleanup.sh"
cat >"$DEACTIVATION_FILE" <<'EOF'
#!/usr/bin/bash

unset LD_PRELOAD
EOF

chmod +x "$DEACTIVATION_FILE"

echo "Finished configuration."
echo
echo "Conda environment: $CONDA_ENV_NAME"
echo "Created files:"
echo "  - $ACTIVATE_FILE"
echo "  - $DEACTIVATION_FILE"
echo
echo "You can now do:"
echo
echo "  conda activate $CONDA_ENV_NAME"
echo
echo "and ROS2 Humble rclpy scripts will be properly working"
echo
echo "Check with:"
echo "   echo \$ROS_DISTRO"
echo "   echo \$LD_LIBRARY_PATH | grep /usr/lib/x86_64-linux-gnu"
