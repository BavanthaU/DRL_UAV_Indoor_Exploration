# Isaac Sim 5.0 Docker Environment

This builds a clean Isaac Sim 5.0.0 + Isaac Lab v2.2.1 image for this project.

Important: this does not replace the host NVIDIA driver. A container launched
with `--gpus all` still uses the host driver mounted by NVIDIA Container
Toolkit. On this machine, Docker sees driver `595.58.03`, so an RTX renderer
crash caused by that driver can still happen inside Docker.

Build:

```bash
docker build -f docker/isaacsim50/Dockerfile -t drl-uav-isaacsim50 .
```

Small test run:

```bash
MAX_ITERATIONS=1 NUM_ENVS=1 docker/isaacsim50/run_train.sh
```

Full run:

```bash
docker/isaacsim50/run_train.sh
```

Useful overrides:

```bash
NUM_ENVS=8 MAX_ITERATIONS=500 DEVICE=cuda:1 docker/isaacsim50/run_train.sh
RENDERING_MODE=balanced docker/isaacsim50/run_train.sh
WANDB_MODE=online docker/isaacsim50/run_train.sh
```
