#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from _vlm_hierarchical_common import (
    build_hierarchical_model,
    build_hierarchical_trainer,
    build_log_dir,
    make_env,
    parse_hierarchical_train_args,
    write_run_config,
)


def main() -> None:
    args, config, simulation_app = parse_hierarchical_train_args()
    log_dir = build_log_dir(config, args)
    model = build_hierarchical_model(config, action_dim=3)
    env = make_env(config, args)
    trainer, max_iterations = build_hierarchical_trainer(config, args, model, log_dir)
    write_run_config(trainer, config)
    if args.checkpoint:
        trainer.load(args.checkpoint)
    result = trainer.train(env, max_iterations=max_iterations)
    checkpoint_path = Path(log_dir) / "checkpoints" / f"hierarchical_policy_update_{result.updates:06d}.pt"
    trainer.save(str(checkpoint_path))
    print(f"[INFO] Hierarchical VLM-PPO training finished: updates={result.updates} timesteps={result.timesteps}")
    print(f"[INFO] Checkpoint saved: {checkpoint_path}")
    trainer.logger.finish()
    if hasattr(env, "close"):
        env.close()
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
