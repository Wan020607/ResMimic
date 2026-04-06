#!/bin/bash
# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Create overall workspace
WORKSPACE_DIR=$SCRIPT_DIR/thirdparty
ENV_NAME=resmimic
SENTINEL_FILE=$WORKSPACE_DIR/.env_setup_finished

ISAACGYM_PATH="/home/wan/snap/IsaacGym_Preview_4_Package/isaacgym"

mkdir -p $WORKSPACE_DIR

# Check if environment setup has already finished
if [[ ! -f $SENTINEL_FILE ]]; then

  # Create the conda environment if it doesn't exist
  if ! conda info --envs | grep -q "$ENV_NAME"; then
      conda create -y -n $ENV_NAME python=3.8
  fi

  # Activate environment
  source $(conda info --base)/etc/profile.d/conda.sh
  conda activate $ENV_NAME

  # Install libstdcxx-ng to fix GLIBCXX_3.4.32 issue
  conda install -c conda-forge -y libstdcxx-ng

  # Install Isaac Gym from existing path
  if [[ -d $ISAACGYM_PATH ]]; then
      cd $ISAACGYM_PATH/python
      pip install -e .
  else
      echo "Error: Isaac Gym path does not exist: $ISAACGYM_PATH"
      exit 1
  fi

  # Return to script directory
  cd $SCRIPT_DIR

  # Install local editable packages
  pip install -e rsl_rl
  pip install -e legged_gym
  pip install -e pose

  # Install other Python packages
  pip install "numpy==1.23.0" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown hydra-core imageio[ffmpeg] mujoco mujoco-python-viewer isaacgym-stubs pytorch-kinematics rich termcolor 
  pip install scipy
  pip install "redis[hiredis]"

  # Install Redis server if not available
  if ! which redis-server; then
      sudo apt install -y redis-server
  fi

  pip install pyttsx3  # for voice control
  pip install trimesh


  # Mark the setup as finished
  touch $SENTINEL_FILE
fi