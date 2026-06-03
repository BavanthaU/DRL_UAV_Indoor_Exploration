# VLM-PPO UAV Indoor Exploration

This branch is focused on one training path:

`Isaac-VLM-PPO-UAV-Exploration-v0`

It contains the code needed to train and evaluate a PPO-based indoor
exploration agent with:

- Isaac Lab direct quadrotor environment,
- continuous body-frame velocity and yaw-rate actions,
- altitude-hold force/torque control,
- VLM camera/map frontend,
- frontier-guided subgoal features,
- new-cell curiosity and optional RND,
- optional offline Qwen2.5-VL teacher labels,
- W&B experiment tracking,
- optional C++ grid-planning acceleration with Python fallback.

Old baseline scripts, old checkpoints, old evaluation CSVs, and unrelated
scaffold code have been removed from this branch. The office USD assets needed
by the new Isaac task are kept under `assets/environments/`.

## Repository Layout

- `assets/environments/` - indoor office USD assets used by the Isaac task.
- `configs/vlm_ppo_explorer/` - debug, RTX 4080, SigLIP, and Jetson/export configs.
- `cpp/grid_planning/` - optional pybind11 extension for grid planning utilities.
- `exploration_stack/tasks/vlm_ppo_exploration/` - Isaac task and debug vector env.
- `exploration_stack/vlm_frontend/` - MobileCLIP/SigLIP/mock VLM policy frontend.
- `exploration_stack/rl/ppo/` - actor-critic, PPO trainer, logging, backend adapters.
- `exploration_stack/vlm_teacher/` - offline Qwen teacher schema and labeler.
- `scripts/` - train, eval, play, profile, rollout collection, export, and labeling.
- `tests/` - focused tests for the VLM-PPO path.

## Setup

Requirements:

- Isaac Sim 5.1.0
- Isaac Lab 2.3.2
- Python 3.11 conda environment for Isaac Lab
- Project dependencies from `requirements.txt`

Isaac Lab environment:

```sh
cd /home/bavantha/IsaacLab
git checkout v2.3.2
./isaaclab.sh -c env_isaaclab
conda activate env_isaaclab
./isaaclab.sh -i
```

Project dependencies:

```sh
cd /home/bavantha/Autonomous_Drone
python -m pip install -r requirements.txt
```

## Debug Smoke Training

This does not launch Isaac Sim. It uses the CPU debug environment and mock VLM
backend only for tests/smoke runs.

```sh
cd /home/bavantha/Autonomous_Drone
conda activate env_isaaclab
python scripts/train_vlm_ppo_explorer.py \
  --config configs/vlm_ppo_explorer/debug_mock_train.yaml \
  --max_iterations 1 \
  --num_envs 2 \
  --wandb_mode disabled
```

## Isaac Lab Training

```sh
cd /home/bavantha/IsaacLab
./isaaclab.sh -p /home/bavantha/Autonomous_Drone/scripts/train_vlm_ppo_explorer.py \
  --task Isaac-VLM-PPO-UAV-Exploration-v0 \
  --config /home/bavantha/Autonomous_Drone/configs/vlm_ppo_explorer/rtx4080_mobileclip_train.yaml \
  --headless \
  --enable_cameras \
  --num_envs 16 \
  --wandb_project vlm-ppo-uav-exploration
```

Use `--wandb_mode disabled` for local-only runs or `--wandb_mode offline` for
offline logging.

## Evaluation

```sh
cd /home/bavantha/IsaacLab
./isaaclab.sh -p /home/bavantha/Autonomous_Drone/scripts/eval_vlm_ppo_explorer.py \
  --task Isaac-VLM-PPO-UAV-Exploration-v0 \
  --config /home/bavantha/Autonomous_Drone/configs/vlm_ppo_explorer/rtx4080_mobileclip_train.yaml \
  --checkpoint /path/to/policy_update_000001.pt \
  --num_eval_episodes 60 \
  --enable_cameras \
  --record_video
```

## Tests

```sh
python -m unittest discover -s tests
/home/bavantha/miniconda3_aiar/envs/env_isaaclab/bin/python -m unittest discover -s tests
```

## Documentation

See `docs/vlm_ppo_curiosity_explorer.md` for configs, W&B tracking, profiling,
rollout collection, teacher labeling, ONNX export, and known limitations.
