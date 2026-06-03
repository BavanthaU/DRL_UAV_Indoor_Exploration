#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch

from _vlm_hierarchical_common import (
    build_hierarchical_model,
    build_hierarchical_trainer,
    build_log_dir,
    make_env,
    parse_hierarchical_train_args,
    write_run_config,
)
from exploration_stack.evaluation import (
    BestExplorationMapTracker,
    save_exploration_map_png,
    write_exploration_map_metadata,
)


def main() -> None:
    args, config, simulation_app = parse_hierarchical_train_args()
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required for evaluation.")
    model = build_hierarchical_model(config, action_dim=3)
    env = make_env(config, args)
    log_dir = build_log_dir(config, args)
    trainer, _ = build_hierarchical_trainer(config, args, model, log_dir)
    write_run_config(trainer, config)
    trainer.load(args.checkpoint)
    obs = trainer._to_device(env.reset())
    episode_returns = torch.zeros(env.num_envs, device=trainer.device)
    episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=trainer.device)
    map_tracker = BestExplorationMapTracker(env)
    finished_returns: list[float] = []
    finished_mapped_cells: list[float] = []
    max_steps = args.steps or int(config.get("environment", {}).get("max_steps", 600)) * args.num_eval_episodes
    last_global_step = 0
    for global_step in range(max_steps):
        last_global_step = global_step + 1
        with torch.no_grad():
            action, info = trainer.model.act(obs, deterministic=True)
        next_obs, reward, done, env_info = env.step(action)
        reward = trainer._as_tensor(reward)
        done = trainer._as_tensor(done).bool()
        episode_returns += reward
        episode_steps += 1
        mapped_cells = env_info.get("reward_terms", {}).get("mapped_free_cells")
        mapped_cells_tensor = trainer._as_tensor(mapped_cells) if mapped_cells is not None else None
        for env_id in torch.nonzero(done, as_tuple=False).flatten().tolist():
            episode_return = float(episode_returns[env_id].detach().cpu())
            snapshot = map_tracker.latest_snapshots[env_id] if env_id < len(map_tracker.latest_snapshots) else None
            mapped_free_cells = _mapped_free_cells_for_env(mapped_cells_tensor, env_id, snapshot)
            finished_returns.append(episode_return)
            finished_mapped_cells.append(mapped_free_cells)
            map_tracker.record_completed(
                env_id=env_id,
                episode_index=len(finished_returns),
                episode_return=episode_return,
                mapped_free_cells=mapped_free_cells,
                episode_steps=int(episode_steps[env_id].detach().cpu()),
                global_step=global_step + 1,
            )
            episode_returns[env_id] = 0.0
            episode_steps[env_id] = 0
            if len(finished_returns) >= args.num_eval_episodes:
                break
        obs = trainer._to_device(next_obs)
        map_tracker.refresh()
        if len(finished_returns) >= args.num_eval_episodes:
            break
    if map_tracker.best is None:
        map_tracker.record_best_partial(episode_returns, episode_steps, global_step=last_global_step)
    metrics = {
        "episodes": len(finished_returns),
        "mean_return": sum(finished_returns) / max(1, len(finished_returns)),
        "mean_mapped_free_cells": sum(finished_mapped_cells) / max(1, len(finished_mapped_cells)),
    }
    output_dir = Path(args.output_dir or log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if map_tracker.best is not None:
        map_dir = output_dir / "eval_maps"
        map_path = save_exploration_map_png(
            map_tracker.best.snapshot,
            map_dir / "best_explored_map.png",
            metadata=map_tracker.best.metadata,
        )
        metadata_path = write_exploration_map_metadata(
            map_tracker.best.snapshot,
            map_dir / "best_explored_map.json",
            extra=map_tracker.best.metadata,
        )
        metrics["best_mapped_free_cells"] = float(map_tracker.best.metadata["mapped_free_cells"])
        metrics["best_episode_return"] = float(map_tracker.best.metadata["episode_return"])
        metrics["best_map_completed"] = float(bool(map_tracker.best.metadata["completed"]))
        trainer.logger.log_image(
            "eval/best_explored_map",
            map_path,
            caption=(
                f"mapped_free_cells={metrics['best_mapped_free_cells']:.0f}, "
                f"return={metrics['best_episode_return']:.3f}"
            ),
            enabled_key="log_eval_maps",
        )
        trainer.logger.log_artifact(
            map_dir,
            artifact_type="eval-map",
            enabled_key="log_eval_maps",
            aliases=["latest", "best"],
            metadata={"map_path": str(map_path), "metadata_path": str(metadata_path)},
        )
    metrics_path = output_dir / "hierarchical_eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    trainer.logger.log(0, {f"eval/{key}": value for key, value in metrics.items()})
    trainer.logger.log_artifact(metrics_path, artifact_type="eval", enabled_key="log_eval", aliases=["latest"])
    trainer.logger.finish()
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if hasattr(env, "close"):
        env.close()
    if simulation_app is not None:
        simulation_app.close()


def _mapped_free_cells_for_env(mapped_cells_tensor, env_id: int, snapshot) -> float:
    fallback = float(snapshot.mapped_free_cells) if snapshot is not None else 0.0
    if mapped_cells_tensor is None:
        return fallback
    if mapped_cells_tensor.numel() == 0:
        return fallback
    if mapped_cells_tensor.ndim == 0 or mapped_cells_tensor.numel() <= env_id:
        return fallback if fallback > 0.0 else float(mapped_cells_tensor.detach().cpu().reshape(-1)[0])
    return float(mapped_cells_tensor[env_id].detach().cpu())


if __name__ == "__main__":
    main()
