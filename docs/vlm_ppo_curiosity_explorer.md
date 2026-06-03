# VLM-PPO Curiosity Explorer

This document describes the retained flat VLM-PPO baseline. The current main
branch method is the learned hierarchical variant documented in
`docs/vlm_hierarchical_ppo_explorer.md` with task id
`Isaac-VLM-Hierarchical-PPO-UAV-Exploration-v0`.

This branch contains a standalone PPO exploration path:

`Isaac-VLM-PPO-UAV-Exploration-v0`

Old baseline source, old checkpoints, old evaluation files, and unrelated
scaffold code have been removed from this branch. The repository now keeps only
the code and assets needed to train, evaluate, profile, export, and label the
VLM-PPO exploration agent.

## What Changed

- Isaac Lab direct task under `exploration_stack/tasks/vlm_ppo_exploration/`.
- Continuous action space: `[v_x_body, v_y_body, yaw_rate]` with altitude hold.
- Crazyflie articulation with force/torque control through Isaac Lab's direct
  environment path.
- Indoor office USD asset spawned by default from `assets/environments/TrainEnvOffice1.usd`.
- VLM frontend under `exploration_stack/vlm_frontend/` with `mobileclip`,
  `siglip`, and `mock` backends.
- PPO actor-critic and trainer adapters under `exploration_stack/rl/ppo/`.
- Curiosity rewards: new-cell count, optional policy-side RND, and semantic
  novelty scaffolding.
- C++/Python grid-planning utilities for A*, frontier extraction, coverage, and
  connected components.
- Offline Qwen2.5-VL teacher labeler for auxiliary supervision only. It is not
  called during PPO rollout.

## Reward Contract

The new task rewards map progress and safety signals:

- New unique free cells discovered.
- Coverage-ratio progress.
- Distance/progress features to selected frontiers.
- Collision, near-obstacle, idle, oscillation, action-smoothness, and altitude
  penalties.
- Success when coverage reaches the configured threshold.

It intentionally avoids predefined region identifiers, fixed coordinate success
regions, and coordinate-specific doorway rewards.

## Configs

- `configs/vlm_ppo_explorer/debug_mock_train.yaml`
  CPU/debug PPO smoke run. Uses `vlm.backend=mock` only for tests.
- `configs/vlm_ppo_explorer/rtx4080_mobileclip_train.yaml`
  Primary RTX 4080 SUPER training config.
- `configs/vlm_ppo_explorer/rtx4080_siglip_train.yaml`
  Higher-quality SigLIP ablation config.
- `configs/vlm_ppo_explorer/jetson_export_student.yaml`
  Student/export config for deployment preparation.
- `configs/vlm_ppo_explorer/jetson_mobileclip_deploy.yaml`
  Jetson-like profiling config.

## Commands

Debug smoke train:

```sh
cd /home/bavantha/Autonomous_Drone
conda activate env_isaaclab
python scripts/train_vlm_ppo_explorer.py \
  --config configs/vlm_ppo_explorer/debug_mock_train.yaml \
  --max_iterations 1 \
  --num_envs 2
```

Isaac Lab training:

```sh
cd /home/bavantha/IsaacLab
./isaaclab.sh -p /home/bavantha/Autonomous_Drone/scripts/train_vlm_ppo_explorer.py \
  --task Isaac-VLM-PPO-UAV-Exploration-v0 \
  --config /home/bavantha/Autonomous_Drone/configs/vlm_ppo_explorer/rtx4080_mobileclip_train.yaml \
  --headless \
  --enable_cameras \
  --num_envs 16
```

Evaluation:

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

Profiling:

```sh
python scripts/profile_vlm_ppo_explorer.py \
  --config configs/vlm_ppo_explorer/debug_mock_train.yaml \
  --num_envs 2
```

Rollout collection:

```sh
python scripts/collect_vlm_ppo_rollouts.py \
  --config configs/vlm_ppo_explorer/debug_mock_train.yaml \
  --steps 128
```

Offline teacher labeling:

```sh
python scripts/label_vlm_affordances_qwen.py \
  --rollout logs/vlm_ppo_explorer/<run_id>/rollouts/rollout.pt \
  --output logs/vlm_ppo_explorer/<run_id>/teacher_labels.jsonl
```

ONNX export:

```sh
python scripts/export_vlm_ppo_policy_onnx.py \
  --config configs/vlm_ppo_explorer/debug_mock_train.yaml \
  --checkpoint /path/to/policy_update_000001.pt
```

## Logging

Runs write to:

`logs/vlm_ppo_explorer/<run_id>/`

Each run stores:

- `config.json`
- `metrics.jsonl`
- `metrics.csv`
- `checkpoints/policy_update_*.pt`

The directory is ignored by git.

## W&B Tracking

The RTX and Jetson configs enable W&B uploads by default under project:

`vlm-ppo-uav-exploration`

Uploaded items:

- scalar PPO metrics and reward terms,
- config plus git commit,
- checkpoints as model artifacts,
- evaluation JSON as eval artifacts,
- profile JSON as profile artifacts,
- ONNX exports as artifacts,
- teacher label JSONL when `scripts/label_vlm_affordances_qwen.py` is run with W&B enabled.

Rollout cache upload is disabled by default because caches can become large.
Enable it by setting:

```json
"wandb": {
  "log_rollouts": true
}
```

Disable W&B for a run:

```sh
python scripts/train_vlm_ppo_explorer.py \
  --config configs/vlm_ppo_explorer/debug_mock_train.yaml \
  --wandb_mode disabled
```

Run in offline mode:

```sh
python scripts/train_vlm_ppo_explorer.py \
  --config configs/vlm_ppo_explorer/debug_mock_train.yaml \
  --wandb_mode offline
```

Override project/entity/name:

```sh
python scripts/train_vlm_ppo_explorer.py \
  --config configs/vlm_ppo_explorer/rtx4080_mobileclip_train.yaml \
  --wandb_project vlm-ppo-uav-exploration \
  --wandb_entity <entity> \
  --wandb_name <run-name>
```

## Verification Run

Commands run during implementation:

```sh
python -m unittest discover -s tests
/home/bavantha/miniconda3_aiar/envs/env_isaaclab/bin/python -m unittest discover -s tests
/home/bavantha/miniconda3_aiar/envs/env_isaaclab/bin/python scripts/train_vlm_ppo_explorer.py --config configs/vlm_ppo_explorer/debug_mock_train.yaml --max_iterations 1 --num_envs 2 --run_id smoke_train
/home/bavantha/miniconda3_aiar/envs/env_isaaclab/bin/python scripts/profile_vlm_ppo_explorer.py --config configs/vlm_ppo_explorer/debug_mock_train.yaml --num_envs 2 --run_id profile_smoke
/home/bavantha/miniconda3_aiar/envs/env_isaaclab/bin/python scripts/collect_vlm_ppo_rollouts.py --config configs/vlm_ppo_explorer/debug_mock_train.yaml --steps 2 --num_envs 2 --run_id collect_smoke
/home/bavantha/miniconda3_aiar/envs/env_isaaclab/bin/python scripts/export_vlm_ppo_policy_onnx.py --config configs/vlm_ppo_explorer/debug_mock_train.yaml --num_envs 1 --run_id export_smoke
```

## Current Limitations

- The real VLM configs require MobileCLIP/open_clip or SigLIP model availability.
- The C++ extension is optional; Python fallback is used automatically when
  `grid_planning_ext` is not installed.
- Building the extension requires `pybind11`.
- The Isaac task must be launched with Isaac Lab's `isaaclab.sh -p` so Omniverse
  modules such as `pxr` are available.
- Semantic image labels from Isaac are not fully plumbed into the direct task yet;
  `semantic_line` is currently present in the observation contract and debug
  environment, while the Isaac direct env returns zeros until semantic camera
  labels are configured for the tiled camera.
