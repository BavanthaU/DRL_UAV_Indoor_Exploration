#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-drl-uav-isaacsim50}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
GPU_FLAG="${GPU_FLAG:---gpus all}"
WANDB_MODE="${WANDB_MODE:-offline}"

docker run --rm -it \
  ${GPU_FLAG} \
  --ipc=host \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}" \
  -e NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}" \
  -e WANDB_MODE="${WANDB_MODE}" \
  -v "${PROJECT_ROOT}:/workspace/project" \
  -v "${HOME}/.cache/ov:/root/.cache/ov" \
  -v "${HOME}/.cache/pip:/root/.cache/pip" \
  -v "${HOME}/.local/share/ov:/root/.local/share/ov" \
  "${IMAGE}" \
  bash -lc '
    set -euo pipefail
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-isaacsim50}"
    export PYTHONPATH=/workspace/project:${PYTHONPATH:-}
    cd /workspace/project
    python scripts/train_depth_hierarchical_joint_rnd.py \
      --config configs/depth_hierarchical_ppo/train_joint_rnd_temporal_isaac.yaml \
      --max_iterations "${MAX_ITERATIONS:-1000}" \
      --num_envs "${NUM_ENVS:-16}" \
      --headless \
      --enable_cameras \
      --rendering_mode "${RENDERING_MODE:-performance}" \
      --device "${DEVICE:-cuda:0}"
  '
