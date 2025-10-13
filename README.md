# IL with RL hybrid for UAV Autonomous Indoor Exploration

## Overview
This repository contains the software developed to enable autonomous indoor exploration with Unmanned Aerial Vehicles (UAVs) using a hybrid Deep Reinforcement Learning (DRL) and Imitation Learning (IL) approach. The system uses Behavior Cloning (BC) to pretrain the UAV from expert demonstrations and then fine-tunes the policy with the Soft Actor-Critic (SAC) DRL algorithm for continuous control. Training and testing are performed in NVIDIA Isaac Sim, a physics-accurate, photorealistic simulator.

The figure below illustrates the methodology designed in this work.

Experiments demonstrate that the hybrid IL+DRL approach converges faster and achieves higher coverage and success rates compared to policies trained with only DRL or IL.

This project is part of a Master’s thesis in Robotics, titled "Pretraining Deep Reinforcement Learning with Imitation Learning for Autonomous UAV Exploration in Unknown Indoor Environments," and is available in the [University of Twente repository](https://purl.utwente.nl/essays/108614).

<img src="./images/Method_schematic.png" width="1000" />

## Folder structure
This repository contains several important directories:

- **`imitation_learning/`** – Contains Python scripts for training imitation learning (IL) algorithms:
  - `IL_train.py` – Trains the IL algorithm using expert demonstrations.
  - `demo_dataset_loader.py` – Loads expert trajectories and feeds them to IL training.
  - `model/` – Defines the neural network architecture used in IL, matching the layers and nodes of the SAC networks used in RL.

- **`isaac45/`** – Contains the main scripts for training and evaluating RL algorithms in Isaac Sim 4.5.0, along with supporting directories required for these scripts:
  - **Main scripts:**
    - `train_exploration.py` – Script to train RL exploration policies.
    - `validate_exploration.py` – Script to evaluate trained IL and RL policies.
  - **Supporting directories required by the scripts:**
    - `RL_drone/` – Environment and agent configurations for training/testing RL algorithms.
    - `drone_models/` – Python configuration files and USD models of drones used in Isaac Lab.
    - `environments/` – USD files of training and testing environments.
    - `mdp/` – Implementation of **MDP components**, including rewards, termination conditions, events, observations, and actions.
    - `utils/` – Utility scripts for RL, including SB3 wrappers for Isaac Lab, RL architectures, and an occupancy map module.

- **`trained_model_and_trajectories/`** – Contains all trained IL and RL models as well as expert trajectories.


## Setup

### Dependencies
- Isaac Sim (4.5.0)
- Isaac Lab (2.1.0)
- Kornia (0.8.1)

### Environment Setting
This project has been developed inside a Docker container on a remote server. 

**0. Download the Base Image of Isaac Sim (Optional)**
Follow the instructions in the [official Nvidia website](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_container.html).

If you are working on the ITC University of Twente server, pulling the Isaac Sim Docker image as described in Step 6 of the Container Deployment section is the only step needed: docker pull nvcr.io/nvidia/isaac-sim:4.5.0

**1. Create the Docker Base Image with Isaac Sim and Isaac Lab:**

Clone the IsaacLab repository and go to the branch corresponding to the version you want to use:
```sh
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab/
git fetch origin
git checkout -b isaaclab-v2.1.0
```

Create an IsaacLab docker image and container. It will ask you to enable X-forwarding, do not enable. Then enter the docker. 
```sh
./docker/container.sh start
./docker/container.sh enter
```

This repository includes large trained IL or RL model files that are stored with **Git LFS**.  
If you want to clone and use the repository correctly, follow these steps:

```sh
apt-get update
apt-get install git-lfs -y
git lfs install
```
Once Git LFS is installed, clone as usual:
```sh
git clone https://github.com/DesireeNP2/DRL_UAV_Indoor_Exploration.git
```

## ⚙️ Usage
### Train IL algorithm with Behavior Cloning
To train IL algorithm run:
```sh
cd DRL_UAV_Indoor_Exploration/imitation_learning
python3 IL_train.py --max_iters 1000 --demo_dir /workspace/isaaclab/DRL_UAV_Indoor_Exploration/trained_model_and_trajectories/expert_trajectories/
```
Depending on your file structure, the flag demo_dir might need a different argument. 


### Train RL algorithms with SAC with or without pretraining.
The GUI can be visualized through the Isaac Sim WebRTC Streaming Client by adding the flag **`--livestream=2`** to any launched command instead of the --headless flag. The streaming client has to be connected through the IP address. Type **`hostname -I`** in terminal to get IP address. 


**With IL pretraining and filling replay buffer**. 
Add path to IL model at the end. You should be in the isaaclab directory. cd **`/workspace/isaaclab`**
```sh
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh \
  -p DRL_UAV_Indoor_Exploration/isaac45/train_exploration.py \
  --enable_cameras \
  --num_envs 10 \
  --headless \
  --task Drone_SAC_IL \
  --use_IL \
  --fill_replay_buffer \
  --IL_model_path /workspace/isaaclab/DRL_UAV_Indoor_Exploration/trained_model_and_trajectories/IL_models_observations/BC_O4.pth
```
Depending on your file structure, the flag IL_model_path might need a different argument. 

**With IL pretraining, buffer already exists**
```sh
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh \
  -p DRL_UAV_Indoor_Exploration/isaac45/train_exploration.py \
  --enable_cameras \
  --num_envs 10 \
  --headless \
  --task Drone_SAC_IL \
  --use_IL \
  --IL_model_path /workspace/isaaclab/DRL_UAV_Indoor_Exploration/trained_model_and_trajectories/IL_models_observations/BC_O4.pth\
  --buffer_path logs/sb3/replay_buffer_widelens.pkl
```

**Without IL pretraining (SAC only)** 
```sh
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh \
  -p DRL_UAV_Indoor_Exploration/isaac45/train_exploration.py \
  --enable_cameras \
  --num_envs 10 \
  --headless \
  --task Drone_SAC_no_IL
```

### Evaluate IL or RL Policies
You can evaluate trained IL or RL policies in different environments. Change the `--task` flag to select the environment you want to evaluate in: `envA`, `envB`, or `envC`.
```sh
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh \
  -p DRL_UAV_Indoor_Exploration/isaac45/validate_exploration.py \
  --enable_cameras \
  --num_envs 1 \
  --headless \
  --task Drone_eval_envA \
  --checkpoint /workspace/isaaclab/DRL_UAV_Indoor_Exploration/trained_model_and_trajectories/RL_models/BC_SAC.zip
```



## Logging during training
Training logs are recorded using TensorBoard. To view them on your local machine while running training on a remote server, follow these steps:

Create an SSH tunnel from your machine to the server:
```sh
ssh -L 6006:localhost:6006 <username>@<servername>
```
Enter the Docker container running Isaac Lab:
```sh
docker exec -it isaac-lab-base bash
```
Launch TensorBoard inside the container:
```sh
tensorboard --logdir ./logs --host localhost --port 6006
```
Open the browser on your local PC and go to:
```sh
http://localhost:6006
```
A different port other than 6006 can be used, if that one is already in use. 

## Changelog

### 2025-03-31
- Allow symmetric `[-1, 1]` velocity commands for drone SAC wrappers so policies can reverse or brake.
- Swapped contact-sensor collision handling for ray-cast proximity rewards/terminations across drone exploration environments.
- Added hierarchical frontier planner support (automatic subgoal selection, subgoal observations, and rewards) for autonomous exploration experiments.
- Introduced frozen MobileNet feature extractor and `sb3_sac_mobilenet.yaml` experiment profile for lightweight pretrained vision backbones.
- Added curiosity-based intrinsic reward derived from cell visit counts to reduce looping behaviour.
- Added planner mode switches (heuristic/observe/rl) with frontier visualisations and subgoal statistics logged to W&B.
- Added configurable map snapshot logging interval to periodically push occupancy/frontier overlays to W&B for debugging.
