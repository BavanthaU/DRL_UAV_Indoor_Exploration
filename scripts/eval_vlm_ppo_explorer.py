#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch

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
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required for evaluation.")
    env_backend = config.get("environment", {}).get("backend", "debug")
    model = build_model(config, action_dim=3)
    env = make_debug_env(config, args) if env_backend == "debug" else make_isaac_env(config, args)
    log_dir = build_log_dir(config, args)
    trainer, _ = build_trainer(config, args, model, log_dir)
    write_run_config(trainer, config)
    trainer.load(args.checkpoint)
    obs = trainer._to_device(env.reset())
    episode_returns = torch.zeros(env.num_envs, device=trainer.device)
    finished_returns: list[float] = []
    finished_mapped_cells: list[float] = []
    max_steps = args.steps or int(config.get("environment", {}).get("max_steps", 600)) * args.num_eval_episodes
    for _ in range(max_steps):
        with torch.no_grad():
            action, _ = trainer.model.act(obs, deterministic=True)
        next_obs, reward, done, info = env.step(action)
        reward = trainer._as_tensor(reward)
        done = trainer._as_tensor(done).bool()
        episode_returns += reward
        mapped_cells = info.get("reward_terms", {}).get("mapped_free_cells")
        mapped_cells_tensor = trainer._as_tensor(mapped_cells) if mapped_cells is not None else torch.zeros(env.num_envs, device=trainer.device)
        for env_id in torch.nonzero(done, as_tuple=False).flatten().tolist():
            finished_returns.append(float(episode_returns[env_id].detach().cpu()))
            finished_mapped_cells.append(float(mapped_cells_tensor[env_id].detach().cpu()))
            episode_returns[env_id] = 0.0
            if len(finished_returns) >= args.num_eval_episodes:
                break
        obs = trainer._to_device(next_obs)
        if len(finished_returns) >= args.num_eval_episodes:
            break
    metrics = {
        "episodes": len(finished_returns),
        "mean_return": sum(finished_returns) / max(1, len(finished_returns)),
        "mean_mapped_free_cells": sum(finished_mapped_cells) / max(1, len(finished_mapped_cells)),
    }
    output_dir = Path(args.output_dir or build_log_dir(config, args))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    trainer.logger.log(0, {f"eval/{key}": value for key, value in metrics.items()})
    trainer.logger.log_artifact(metrics_path, artifact_type="eval", enabled_key="log_eval", aliases=["latest"])
    trainer.logger.finish()
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if hasattr(env, "close"):
        env.close()
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
