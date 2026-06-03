#!/usr/bin/env python3
from __future__ import annotations

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


def main() -> None:
    args, config, simulation_app = parse_hierarchical_train_args()
    env = make_env(config, args)
    model = build_hierarchical_model(config, action_dim=3)
    log_dir = build_log_dir(config, args)
    trainer, _ = build_hierarchical_trainer(config, args, model, log_dir)
    write_run_config(trainer, config)
    if args.checkpoint:
        trainer.load(args.checkpoint)
    obs = trainer._to_device(env.reset())
    steps = args.steps or int(config.get("rollout_collection", {}).get("steps", 128))
    output_dir = Path(args.output_dir or log_dir / "rollouts")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for _ in range(steps):
        with torch.no_grad():
            action, info = trainer.model.act(obs, deterministic=False)
        next_obs, reward, done, env_info = env.step(action.detach())
        frames.append(
            {
                "obs": {key: value.detach().cpu() for key, value in obs.items()},
                "action": action.detach().cpu(),
                "option": info["option"].detach().cpu(),
                "candidate": info["candidate"].detach().cpu(),
                "option_logits": info["aux"]["option_logits"].detach().cpu(),
                "candidate_logits": info["aux"]["candidate_logits"].detach().cpu(),
                "candidate_mask": info["aux"]["candidate_mask"].detach().cpu(),
                "reward": trainer._as_tensor(reward).detach().cpu(),
                "done": trainer._as_tensor(done).detach().cpu(),
                "prompt_similarity": info["aux"].get("prompt_similarity", torch.empty(0)).detach().cpu(),
                "reward_terms": {
                    key: trainer._as_tensor(value).detach().cpu()
                    for key, value in env_info.get("reward_terms", {}).items()
                    if hasattr(value, "shape") or isinstance(value, (int, float, bool))
                },
            }
        )
        obs = trainer._to_device(next_obs)
    output_path = output_dir / "hierarchical_rollout.pt"
    torch.save({"frames": frames, "config": config}, output_path)
    trainer.logger.log(0, {"rollout/steps": steps, "rollout/num_envs": env.num_envs})
    trainer.logger.log_artifact(output_path, artifact_type="rollout", enabled_key="log_rollouts", aliases=["latest"])
    trainer.logger.finish()
    print(f"[INFO] Saved hierarchical rollout cache: {output_path}")
    if hasattr(env, "close"):
        env.close()
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
