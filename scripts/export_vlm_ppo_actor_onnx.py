#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import torch

from _vlm_ppo_common import build_log_dir, build_model, build_trainer, make_debug_env, parse_train_args, write_run_config


class ActorOnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, camera_rgb, map_crop, frontier_mask, trajectory_mask, depth_line, semantic_line, subgoal_features):
        obs = {
            "camera_rgb": camera_rgb,
            "map_crop": map_crop,
            "frontier_mask": frontier_mask,
            "trajectory_mask": trajectory_mask,
            "depth_line": depth_line,
            "semantic_line": semantic_line,
            "subgoal_features": subgoal_features,
        }
        return self.model.forward(obs).mean


def main() -> None:
    args, config, _ = parse_train_args()
    env = make_debug_env(config, args)
    model = build_model(config, action_dim=3)
    log_dir = build_log_dir(config, args)
    trainer, _ = build_trainer(config, args, model, log_dir)
    write_run_config(trainer, config)
    if args.checkpoint:
        trainer.load(args.checkpoint)
    obs = trainer._to_device(env.reset())
    wrapper = ActorOnnxWrapper(trainer.model).eval()
    output_dir = Path(args.output_dir or log_dir / "exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vlm_ppo_actor.onnx"
    keys = ["camera_rgb", "map_crop", "frontier_mask", "trajectory_mask", "depth_line", "semantic_line", "subgoal_features"]
    inputs = tuple(obs[key] for key in keys)
    with torch.no_grad():
        wrapper(*inputs)
    torch.onnx.export(
        wrapper,
        inputs,
        output_path,
        input_names=keys,
        output_names=["action_mean"],
        dynamic_axes={name: {0: "batch"} for name in [*keys, "action_mean"]},
        opset_version=17,
    )
    trainer.logger.log_artifact(output_path, artifact_type="onnx", enabled_key="log_exports", aliases=["latest"])
    trainer.logger.finish()
    print(f"[INFO] Exported ONNX actor: {output_path}")


if __name__ == "__main__":
    main()
