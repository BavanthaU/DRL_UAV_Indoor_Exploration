# Four-Layer UAV Exploration Architecture

This repository now keeps the existing IL/SAC Isaac training path and adds a
parallel four-layer exploration stack for the next research step. The new stack
is designed to support no-demo global exploration, proper SLAM/mapping, and
low-rate semantic reasoning without deleting the working policy code.

## Layer 1: Robot, Sensor, And Control

Files:
- `exploration_stack/core/types.py`
- `exploration_stack/robot/base.py`
- `exploration_stack/robot/existing_repo_adapter.py`
- `exploration_stack/robot/isaaclab_quadcopter_adapter.py`
- `configs/robot/*.yaml`

`RobotAdapter` exposes:
- `reset()`
- `step(action)`
- `get_state()`
- `get_sensors()`
- `apply_local_command(cmd)`
- `close()`

`ExistingRepoRobotAdapter` wraps the current Isaac/Gym/SB3-style environment so
the old IL/SAC policy remains callable. `IsaacLabQuadcopterAdapter` is a clear
skeleton for replacing the current ideal 2D action term with an Isaac Lab
quadcopter task/template. It raises an explicit dependency or implementation
error instead of silently falling back to the old controller.

## Layer 2: SLAM And Map Memory

Files:
- `exploration_stack/slam/*`
- `exploration_stack/mapping/*`
- `configs/slam/isaac_ros_visual_slam.yaml`
- `configs/mapping/nvblox.yaml`

`SlamBackend` and `MappingBackend` separate pose estimation from map memory.
The current implementation includes:
- `SimGroundTruthSlamBackend`: debug/smoke-test only; logs a warning if selected
  for training.
- `IsaacRosVisualSlamBackend`: ROS 2 / Isaac ROS Visual SLAM shell.
- `MapFromExistingOccupancyBackend`: fallback around the current occupancy map.
- `NvbloxMappingBackend` and `RtabmapMappingBackend`: ROS 2 wrapper shells.

Pure-Python map utilities provide:
- occupancy normalization for existing-repo and ROS-style grids,
- frontier extraction,
- frontier clustering,
- coverage calculation,
- 3D occupancy to 2D projection,
- ESDF-to-safe-grid conversion,
- lightweight room graph updates from doorway hypotheses.

Ground-truth pose or existing occupancy should be used only for smoke tests,
debugging, or evaluation baselines until proper SLAM/mapping is wired in.

## Layer 3: Semantic VLM/LLM Intuition

Files:
- `exploration_stack/semantic/*`
- `prompts/vlm_frontier_scoring.md`
- `prompts/llm_frontier_scoring.md`
- `configs/semantic/*.yaml`
- `scripts/label_semantic_priors_with_vlm.py`
- `scripts/train_semantic_prior_net.py`
- `scripts/export_semantic_prior_net_onnx.py`

The semantic layer scores frontiers and produces a `SemanticPrior`.

Implemented now:
- `SemanticHeuristicReasoner`: dependency-free baseline using frontier size,
  depth-line openness, semantic-line hints, frontier shape, and ESDF risk.
- `VlmSemanticReasoner`: optional offline/low-rate provider with strict JSON
  validation and cache.
- `LlmMapReasoner`: structured-map-only provider shell.
- `SemanticPriorNet`: small PyTorch model skeleton for later distillation.

The VLM path is intentionally low-rate/offline. A 7B VLM such as Qwen2.5-VL-7B
can label rollouts on the RTX 4080 SUPER, but it should not run inside the
policy-rate controller. Jetson deployment should use a distilled
`SemanticPriorNet`, heuristics, or an optional small VLM at very low rate.

## Layer 4: Hierarchical Global-Local Exploration

Files:
- `exploration_stack/planning/*`
- `exploration_stack/control/*`
- `exploration_stack/runtime/four_layer_runner.py`
- `configs/hierarchical/*.yaml`

The global planner runs at low rate and selects semantic frontiers. The local
controller runs at policy rate and moves toward the selected subgoal. The safety
shield filters every command.

Implemented now:
- A* path planning over the normalized 2D grid.
- `FrontierGraphGlobalExplorer` with information gain, semantic score, doorway
  likelihood, corridor likelihood, path cost, risk, revisit penalty, and loop
  penalty terms.
- `SubgoalManager` for replan/reached/stale logic.
- `ExistingILSACLocalController` wrapper for the old policy, with a simple
  subgoal fallback for smoke tests.
- `EsdfSafetyShield` for ESDF/clearance-based command intervention.
- `FourLayerRunner` orchestration with dependency injection.

Goal-conditioning the IL/SAC policy is not implemented in this first task. The
wrapper leaves the insertion point explicit: add local vector-to-subgoal,
heading, distance, ESDF clearance, and existing depth/semantic line inputs when
the policy is retrained.

## Data Flow

At each runner step:

1. Robot adapter returns `SensorPacket` and `DroneState`.
2. SLAM backend updates `SlamState`.
3. Mapping backend updates `ExplorationMap`.
4. Frontiers are extracted from `OccupancyGrid2D`.
5. Subgoal manager decides whether to replan.
6. Semantic reasoner scores frontier candidates.
7. Global planner selects a `Subgoal`.
8. Local controller outputs `LocalCommand`.
9. Safety shield filters the command.
10. Robot adapter applies or steps the command.

## Verification

Run the no-Isaac smoke test:

```sh
python scripts/run_four_layer_smoke_test.py --steps 5
```

Run unit tests:

```sh
python -m unittest discover -s tests
```

Run import/compile validation:

```sh
python -m compileall -q exploration_stack scripts tests
python -m compileall -q isaac45 imitation_learning
```

These checks do not require Isaac Sim, ROS 2, or VLM dependencies.

## Reward Integrity

The new RL path must not use hard-coded room IDs, doorway coordinates, or
scene-specific coordinate ranges as reward. Use `Drone_SAC_no_IL_MapProgress_V1`
for new no-IL training. Historical thesis tasks remain registered only for
reproduction/comparison. See `docs/reward_design.md`.

## Remaining Work

Immediate implementation:
- Wire `IsaacLabQuadcopterAdapter` to Isaac Lab's quadcopter task/template.
- Replace debug SLAM with Isaac ROS Visual SLAM / cuVSLAM topics.
- Replace existing occupancy fallback with Nvblox occupancy/ESDF output.
- Save rollout packets for offline semantic labeling.
- Add the tensorization/training loop for `SemanticPriorNet`.

Medium-term research:
- Goal-condition the local SAC policy on selected frontier/subgoal features.
- Add no-demo intrinsic rewards based on information gain, frontier progress,
  revisit penalties, and map prediction uncertainty.
- Benchmark old IL/SAC, SAC-only, frontier heuristic, and four-layer semantic
  planner across train and held-out office environments.
