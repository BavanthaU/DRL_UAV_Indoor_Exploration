#!/usr/bin/env python3
from __future__ import annotations

import torch

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
    env = make_env(config, args)
    model = build_hierarchical_model(config, action_dim=3)
    trainer, _ = build_hierarchical_trainer(config, args, model, build_log_dir(config, args))
    write_run_config(trainer, config)
    if args.checkpoint:
        trainer.load(args.checkpoint)
    obs = trainer._to_device(env.reset())
    returns = torch.zeros(env.num_envs, device=trainer.device)
    max_steps = args.steps or int(config.get("environment", {}).get("max_steps", 600))
    last_option = None
    for step in range(max_steps):
        with torch.no_grad():
            action, info = trainer.model.act(obs, deterministic=True)
        last_option = info["option"]
        next_obs, reward, done, _ = env.step(action)
        returns += trainer._as_tensor(reward)
        obs = trainer._to_device(next_obs)
        if trainer._as_tensor(done).bool().any():
            break
    option_mean = float(last_option.float().mean().detach().cpu()) if last_option is not None else -1.0
    trainer.logger.log(0, {"play/steps": step + 1, "play/mean_return": float(returns.mean().detach().cpu())})
    trainer.logger.finish()
    print(
        f"[INFO] Hierarchical play finished after {step + 1} steps. "
        f"Mean return={returns.mean().item():.3f} mean option id={option_mean:.2f}"
    )
    if hasattr(env, "close"):
        env.close()
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
