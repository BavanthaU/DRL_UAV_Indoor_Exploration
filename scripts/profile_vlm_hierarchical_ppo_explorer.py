#!/usr/bin/env python3
from __future__ import annotations

import json
import time
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
from exploration_stack.cpp_accel import grid_planning


def _ms(fn, repeats: int = 1):
    start = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = fn()
    elapsed = (time.perf_counter() - start) * 1000.0 / max(1, repeats)
    return elapsed, result


def main() -> None:
    args, config, simulation_app = parse_hierarchical_train_args()
    env = make_env(config, args)
    model = build_hierarchical_model(config, action_dim=3)
    log_dir = build_log_dir(config, args)
    trainer, _ = build_hierarchical_trainer(config, args, model, log_dir)
    write_run_config(trainer, config)
    obs = trainer._to_device(env.reset())
    sample_grid = [[1] * 32 for _ in range(32)]
    for idx in range(4, 28):
        sample_grid[16][idx] = 2
    sample_grid[16][16] = 1
    unknown_grid = [[0 if r in (0, 31) or c in (0, 31) else sample_grid[r][c] for c in range(32)] for r in range(32)]
    goal_mask = [[False] * 32 for _ in range(32)]
    goal_mask[25][25] = True
    astar_ms, _ = _ms(lambda: grid_planning.astar_to_any_goal((2, 2), goal_mask, sample_grid), repeats=5)
    frontier_ms, _ = _ms(lambda: grid_planning.extract_frontiers(unknown_grid, 0, 1, 2), repeats=5)
    vlm_image_ms, _ = _ms(lambda: model.encoder.vlm.encode_image(obs["camera_rgb"]), repeats=3)
    text = model.encoder.prompt_bank.prompts
    vlm_text_ms, text_embeddings = _ms(lambda: model.encoder.vlm.encode_text(text), repeats=3)
    image_embedding = model.encoder.vlm.encode_image(obs["camera_rgb"])
    prompt_ms, _ = _ms(lambda: image_embedding @ text_embeddings.to(image_embedding.device).T, repeats=10)
    candidate_ms, _ = _ms(lambda: model.candidate_builder.build(obs), repeats=5)
    policy_ms, _ = _ms(lambda: trainer.model.forward(obs), repeats=3)
    action, _ = trainer.model.act(obs)
    step_ms, _ = _ms(lambda: env.step(action.detach()), repeats=3)
    ppo_ms, _ = _ms(lambda: trainer.train(env, max_iterations=1), repeats=1)
    gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    profile = {
        "cpp_accel_active": grid_planning.CPP_ACCEL_ACTIVE,
        "mapping_update_time_ms": step_ms,
        "cpp_astar_time_ms": astar_ms,
        "frontier_extraction_time_ms": frontier_ms,
        "vlm_image_encoder_time_ms": vlm_image_ms,
        "vlm_text_prompt_time_ms": vlm_text_ms + prompt_ms,
        "candidate_build_time_ms": candidate_ms,
        "hierarchical_policy_forward_time_ms": policy_ms,
        "ppo_update_time_ms": ppo_ms,
        "gpu_memory_mb": gpu_memory_mb,
        "achieved_policy_hz": 1000.0 / max(policy_ms, 1.0e-6),
    }
    output_dir = Path(args.output_dir or log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "hierarchical_profile_metrics.json"
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    trainer.logger.log(0, {f"profile/{key}": value for key, value in profile.items()})
    trainer.logger.log_artifact(profile_path, artifact_type="profile", enabled_key="log_profile", aliases=["latest"])
    trainer.logger.finish()
    print(json.dumps(profile, indent=2, sort_keys=True))
    if hasattr(env, "close"):
        env.close()
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
