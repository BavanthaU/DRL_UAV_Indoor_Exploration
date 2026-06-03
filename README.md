# Learned Hierarchical VLM-PPO UAV Indoor Exploration

This branch is focused on the learned hierarchical RL training path:

`Isaac-VLM-Hierarchical-PPO-UAV-Exploration-v0`

The method is not a classical `SLAM -> frontier planner -> A* -> local controller`
pipeline. The main training path is:

`map memory -> VLM affordance memory -> learned semantic option policy -> learned local control policy`

Classical frontier and A* utilities are retained as proposal/features, dropout
ablations, and evaluation baselines. They do not directly command the UAV in the
main hierarchical PPO trainer.

## What Is Included

- Isaac Lab 2.3.2 / Isaac Sim 5.1 compatible quadrotor exploration task.
- Separate hierarchical task package under
  `exploration_stack/tasks/vlm_hierarchical_ppo_exploration/`.
- Learned option set:
  `EXPLORE_FRONTIER_CLUSTER`, `ENTER_DOORWAY`, `FOLLOW_CORRIDOR`,
  `SWEEP_OPEN_SPACE`, `BACKTRACK_TO_UNVISITED_BRANCH`, `ROTATE_SCAN`,
  `AVOID_AND_RECOVER`, `STOP_IF_COMPLETE`.
- Learned continuous local action head for `[vx_body, vy_body, yaw_rate]`.
- VLM prompt bank with exploration affordances, including
  `"an area that should be explored next"`.
- Planner feature modes: `none`, `features_only`, `proposal_only`,
  `oracle_baseline`.
- Planner dropout defaults for the main run:
  `planner_dropout_prob=0.2`, `astar_feature_dropout_prob=0.2`,
  `frontier_candidate_dropout_prob=0.1`.
- Offline Qwen2.5-VL labeler for auxiliary labels only. Qwen is not called
  during PPO rollout.
- W&B logging for configs, metrics, checkpoints, eval/profile artifacts, and
  optional rollout/teacher-label artifacts.

## Repository Layout

- `assets/environments/` - indoor office USD assets used by the Isaac task.
- `configs/vlm_hierarchical_ppo/` - B0-B8 hierarchy configs and debug smoke config.
- `configs/vlm_ppo_explorer/` - flat VLM-PPO baseline configs.
- `exploration_stack/hierarchy/` - candidate builder, option policy, option
  masking, actor-critic, and option credit utilities.
- `exploration_stack/planning/` - planner-derived features, frontier candidates,
  and dropout.
- `exploration_stack/rl/ppo/` - flat and hierarchical PPO trainers.
- `exploration_stack/tasks/vlm_hierarchical_ppo_exploration/` - hierarchical task id.
- `exploration_stack/tasks/vlm_ppo_exploration/` - flat task and CPU debug env.
- `exploration_stack/vlm_frontend/` - MobileCLIP/SigLIP/mock VLM policy frontend.
- `exploration_stack/vlm_teacher/` - offline Qwen teacher schema and labeler.
- `scripts/` - train, eval, play, profile, rollout collection, export, and labeling.
- `tests/` - focused tests for hierarchy, configs, frontend, planning, and smoke training.

## Setup

Requirements:

- Isaac Sim 5.1.0
- Isaac Lab 2.3.2
- Python 3.11 conda environment for Isaac Lab
- Project dependencies from `requirements.txt`

```sh
cd /home/bavantha/IsaacLab
git checkout v2.3.2
./isaaclab.sh -c env_isaaclab
conda activate env_isaaclab
./isaaclab.sh -i

cd /home/bavantha/Autonomous_Drone
python -m pip install -r requirements.txt
```

## Debug Smoke Training

This does not launch Isaac Sim. It uses the CPU debug environment and mock VLM
backend only for tests/smoke runs.

```sh
cd /home/bavantha/Autonomous_Drone
conda activate env_isaaclab
python scripts/train_vlm_hierarchical_ppo_explorer.py \
  --config configs/vlm_hierarchical_ppo/debug_mock.yaml \
  --max_iterations 1 \
  --num_envs 2 \
  --wandb_mode disabled
```

## Main Isaac Training

```sh
cd /home/bavantha/IsaacLab
./isaaclab.sh -p /home/bavantha/Autonomous_Drone/scripts/train_vlm_hierarchical_ppo_explorer.py \
  --task Isaac-VLM-Hierarchical-PPO-UAV-Exploration-v0 \
  --config /home/bavantha/Autonomous_Drone/configs/vlm_hierarchical_ppo/planner_dropout_main.yaml \
  --headless \
  --enable_cameras \
  --num_envs 16 \
  --wandb_project vlm-hierarchical-ppo-uav-exploration
```

Use `--wandb_mode disabled` for local-only runs or `--wandb_mode offline` for
offline W&B logging.

## Tests

```sh
python -m compileall -q exploration_stack scripts tests
python -m unittest discover -s tests
/home/bavantha/miniconda3_aiar/envs/env_isaaclab/bin/python -m unittest discover -s tests
```

## Documentation

See `docs/vlm_hierarchical_ppo_explorer.md` for method details, baselines,
commands, W&B tracking, verification, and current limitations.
