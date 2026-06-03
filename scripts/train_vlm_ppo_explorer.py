#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from _vlm_ppo_common import (
    build_log_dir,
    build_model,
    build_trainer,
    make_debug_env,
    make_isaac_env,
    parse_train_args,
    write_run_config,
)


def main() -> None:
    args, config, simulation_app = parse_train_args()
    log_dir = build_log_dir(config, args)
    env_backend = config.get("environment", {}).get("backend", "debug")
    env = make_debug_env(config, args) if env_backend == "debug" else make_isaac_env(config, args)
    model = build_model(config, action_dim=3)
    trainer, max_iterations = build_trainer(config, args, model, log_dir)
    write_run_config(trainer, config)
    if args.checkpoint:
        trainer.load(args.checkpoint)
    result = trainer.train(env, max_iterations=max_iterations)
    checkpoint_path = Path(log_dir) / "checkpoints" / f"policy_update_{result.updates:06d}.pt"
    trainer.save(str(checkpoint_path))
    print(f"[INFO] VLM-PPO training finished: updates={result.updates} timesteps={result.timesteps}")
    print(f"[INFO] Checkpoint saved: {checkpoint_path}")
    if hasattr(env, "close"):
        env.close()
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
