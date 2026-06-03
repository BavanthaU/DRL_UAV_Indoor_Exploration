#!/usr/bin/env python3
from __future__ import annotations

import torch

from _vlm_ppo_common import build_log_dir, build_model, build_trainer, make_debug_env, make_isaac_env, parse_train_args


def main() -> None:
    args, config, simulation_app = parse_train_args()
    env_backend = config.get("environment", {}).get("backend", "debug")
    env = make_debug_env(config, args) if env_backend == "debug" else make_isaac_env(config, args)
    model = build_model(config, action_dim=3)
    trainer, _ = build_trainer(config, args, model, build_log_dir(config, args))
    if args.checkpoint:
        trainer.load(args.checkpoint)
    obs = trainer._to_device(env.reset())
    returns = torch.zeros(env.num_envs, device=trainer.device)
    max_steps = args.steps or int(config.get("environment", {}).get("max_steps", 600))
    for step in range(max_steps):
        with torch.no_grad():
            action, _ = trainer.model.act(obs, deterministic=True)
        obs_next, reward, done, info = env.step(action)
        returns += trainer._as_tensor(reward)
        obs = trainer._to_device(obs_next)
        if trainer._as_tensor(done).bool().any():
            break
    print(f"[INFO] Play finished after {step + 1} steps. Mean return={returns.mean().item():.3f}")
    if hasattr(env, "close"):
        env.close()
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
