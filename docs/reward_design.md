# Reward Design And Map-Progress Training

The old thesis-era training config used hard-coded doorway coordinates as a
reward term. That made the MDP partly privileged: the agent could receive reward
for crossing locations that came from environment annotations rather than from
its own sensing and mapping.

For new research experiments, use:

```sh
--task Drone_SAC_no_IL_MapProgress_V1
```

## Allowed Training Signals

The map-progress task uses:
- incremental online map coverage,
- loop/no-progress penalty,
- collision penalty from contact sensing,
- idle penalty from action magnitude,
- generic fixed-area termination based on explored cells.

These are acceptable because they are derived from the same online map/safety
state used by the agent or from simulator contact events needed to define
episode failure.

## Disallowed Training Signals

Do not use these as RL rewards for new experiments:
- hard-coded room IDs,
- hard-coded doorway positions,
- scene-specific coordinate ranges,
- manually marked target rooms,
- semantic object labels that are unavailable to the deployed perception stack,
- ground-truth pose or map except for debug/evaluation.

Doorway and room concepts may still exist as semantic hypotheses in the
four-layer planner, but they must come from perception, SLAM/map structure, or
offline VLM labels that are later distilled into a deployable model. They must
not be simulator-provided reward shortcuts.

## Current Task Split

Historical tasks such as `Drone_SAC_no_IL_V1` and `Drone_SAC_IL_V1` are kept so
old results can be reproduced and compared. They are not the recommended path
for new no-imitation-learning work.

Recommended training task:
- `Drone_SAC_no_IL_MapProgress_V1`

Recommended smoke test:

```sh
python scripts/run_four_layer_smoke_test.py --steps 5
python -m unittest discover -s tests
```
