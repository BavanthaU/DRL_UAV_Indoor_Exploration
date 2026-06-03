# Learned Hierarchical VLM-PPO Explorer

## Method

The main method is a learned four-level hierarchy:

1. Map and memory layer from the agent observation history.
2. VLM affordance memory from camera/map renderings and prompt similarities.
3. Learned semantic option policy over exploration options.
4. Learned continuous local controller for `[vx_body, vy_body, yaw_rate]`.

This is intentionally different from a hard-coded four-level planner. Frontier
and A* code is kept for candidate proposals, planner-derived features,
dropout/ablation, and `oracle_baseline` evaluation. The main `features_only`
training config does not send planner actions to the environment.

## Reward Contract

Rewards are based on actual exploration progress and safety:

- unique newly observed cells,
- local map progress from cells observed by the onboard mapping state,
- completion when the local map has no remaining frontier/opening candidates for
  the configured patience window,
- return-to-start progress after the mission reserve time begins,
- collision, near-obstacle, idle, oscillation, action-smoothness, and altitude penalties,
- optional intrinsic terms from the existing flat debug/Isaac task.

The task does not receive predefined region identifiers, fixed coordinate success
regions, target area percentages, known free-cell totals, or coordinate-specific
entry rewards. For a 10 minute mission, the main configs switch to return-home
behavior after 480 seconds so the last 2 minutes are reserved for recovery.

## Main Config

Use B7 for the main experiment:

`configs/vlm_hierarchical_ppo/planner_dropout_main.yaml`

Important defaults:

- `planner_feature_mode: "features_only"`
- `use_privileged_map_for_training: false`
- `planner_dropout_prob: 0.2`
- `astar_feature_dropout_prob: 0.2`
- `frontier_candidate_dropout_prob: 0.1`
- `return_home_after_s: 480.0`

## Baselines and Ablations

- B0: `classical_frontier_baseline.yaml`
- B1: `flat_vlm_ppo_baseline.yaml`
- B2: `no_vlm_hierarchical_baseline.yaml`
- B3: `no_astar_features_ablation.yaml`
- B4: `qwen_auxiliary_ablation.yaml`
- B5: `jetson_deploy_student.yaml`
- B7: `planner_dropout_main.yaml`
- B8: `rtx4080_siglip_ablation.yaml`

## Commands

Debug smoke train:

```sh
python scripts/train_vlm_hierarchical_ppo_explorer.py \
  --config configs/vlm_hierarchical_ppo/debug_mock.yaml \
  --max_iterations 1 \
  --num_envs 2 \
  --wandb_mode disabled
```

Isaac Lab train:

```sh
cd /home/bavantha/IsaacLab
./isaaclab.sh -p /home/bavantha/Autonomous_Drone/scripts/train_vlm_hierarchical_ppo_explorer.py \
  --task Isaac-VLM-Hierarchical-PPO-UAV-Exploration-v0 \
  --config /home/bavantha/Autonomous_Drone/configs/vlm_hierarchical_ppo/planner_dropout_main.yaml \
  --headless \
  --enable_cameras \
  --num_envs 16
```

Evaluate:

```sh
python scripts/eval_vlm_hierarchical_ppo_explorer.py \
  --config configs/vlm_hierarchical_ppo/debug_mock.yaml \
  --checkpoint logs/vlm_hierarchical_ppo/<run_id>/checkpoints/hierarchical_policy_update_000001.pt \
  --num_eval_episodes 10 \
  --wandb_mode disabled
```

Collect rollouts:

```sh
python scripts/collect_vlm_hierarchical_ppo_rollouts.py \
  --config configs/vlm_hierarchical_ppo/debug_mock.yaml \
  --steps 128 \
  --wandb_mode disabled
```

Offline Qwen labels:

```sh
python scripts/label_vlm_hierarchical_rollouts_qwen.py \
  --rollout logs/vlm_hierarchical_ppo/<run_id>/rollouts/hierarchical_rollout.pt \
  --output logs/vlm_hierarchical_ppo/<run_id>/teacher_labels.jsonl
```

## Logging

Runs write to:

`logs/vlm_hierarchical_ppo/<run_id>/`

Each run stores `config.json`, `metrics.jsonl`, `metrics.csv`, and model
checkpoints. Real configs enable W&B uploads by default under:

`vlm-hierarchical-ppo-uav-exploration`

## Verification

Expected local checks:

```sh
python -m compileall -q exploration_stack scripts tests
python -m unittest discover -s tests
python scripts/train_vlm_hierarchical_ppo_explorer.py --config configs/vlm_hierarchical_ppo/debug_mock.yaml --max_iterations 1 --num_envs 2 --wandb_mode disabled
```

## Current Limitations

- The debug environment is a CPU smoke environment, not Isaac physics.
- The reference PPO adapter supports interval-based option holding through
  `option_interval_steps`, but the option policy and local controller are still
  optimized jointly in one PPO loss.
- Real MobileCLIP/SigLIP runs require model dependencies and Isaac Lab launch
  through `isaaclab.sh -p`.
