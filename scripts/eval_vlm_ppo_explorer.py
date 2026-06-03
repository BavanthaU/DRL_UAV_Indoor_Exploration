#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch

from _vlm_ppo_common import build_log_dir, build_model, build_trainer, make_debug_env, make_isaac_env, parse_train_args


def main() -> None:
    args, config, simulation_app = parse_train_args()
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required for evaluation.")
    env_backend = config.get("environment", {}).get("backend", "debug")
    env = make_debug_env(config, args) if env_backend == "debug" else make_isaac_env(config, args)
    model = build_model(config, action_dim=3)
    trainer, _ = build_trainer(config, args, model, build_log_dir(config, args))
    trainer.load(args.checkpoint)
    obs = trainer._to_device(env.reset())
    episode_returns = torch.zeros(env.num_envs, device=trainer.device)
    finished_returns: list[float] = []
    finished_coverages: list[float] = []
    max_steps = args.steps or int(config.get("environment", {}).get("max_steps", 600)) * args.num_eval_episodes
    for _ in range(max_steps):
        with torch.no_grad():
            action, _ = trainer.model.act(obs, deterministic=True)
        next_obs, reward, done, info = env.step(action)
        reward = trainer._as_tensor(reward)
        done = trainer._as_tensor(done).bool()
        episode_returns += reward
        coverage = info.get("reward_terms", {}).get("coverage")
        coverage_tensor = trainer._as_tensor(coverage) if coverage is not None else torch.zeros(env.num_envs, device=trainer.device)
        for env_id in torch.nonzero(done, as_tuple=False).flatten().tolist():
            finished_returns.append(float(episode_returns[env_id].detach().cpu()))
            finished_coverages.append(float(coverage_tensor[env_id].detach().cpu()))
            episode_returns[env_id] = 0.0
            if len(finished_returns) >= args.num_eval_episodes:
                break
        obs = trainer._to_device(next_obs)
        if len(finished_returns) >= args.num_eval_episodes:
            break
    metrics = {
        "episodes": len(finished_returns),
        "mean_return": sum(finished_returns) / max(1, len(finished_returns)),
        "mean_coverage": sum(finished_coverages) / max(1, len(finished_coverages)),
    }
    output_dir = Path(args.output_dir or build_log_dir(config, args))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if hasattr(env, "close"):
        env.close()
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
