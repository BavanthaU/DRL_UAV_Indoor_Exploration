#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exploration_stack.vlm_teacher.qwen25vl_teacher import Qwen25VLTeacher, QwenTeacherConfig, load_teacher_prompt


def _tensor_to_pil(tensor) -> Image.Image:
    image = tensor.detach().cpu().float()
    if image.ndim == 4:
        image = image[0]
    if image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    image = image.clamp(0.0, 1.0)
    array = (image.numpy() * 255.0).astype("uint8")
    return Image.fromarray(array)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label saved VLM-PPO rollout frames with offline Qwen2.5-VL.")
    parser.add_argument("--rollout", required=True, help="Path to rollout.pt from collect_vlm_ppo_rollouts.py")
    parser.add_argument("--output", required=True, help="Output JSONL label path")
    parser.add_argument("--prompt", default="prompts/qwen_exploration_affordance_labeling.md")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--fallback_model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--wandb_project", default="vlm-ppo-uav-exploration")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_name", default=None)
    args = parser.parse_args()

    rollout = torch.load(args.rollout, map_location="cpu")
    frames = rollout.get("frames", rollout)
    prompt = load_teacher_prompt(args.prompt)
    teacher = Qwen25VLTeacher(QwenTeacherConfig(model_name=args.model, fallback_model_name=args.fallback_model))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for frame_id, frame in enumerate(frames[: args.max_frames]):
            image = _tensor_to_pil(frame["obs"]["camera_rgb"])
            label = teacher.label_image(image, prompt)
            file.write(json.dumps({"frame_id": frame_id, "label": label.__dict__}, sort_keys=True) + "\n")
    if args.wandb_mode != "disabled":
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("Install wandb or use --wandb_mode disabled.") from exc
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or output.stem,
            mode=args.wandb_mode,
            config={"rollout": args.rollout, "model": args.model, "fallback_model": args.fallback_model},
        )
        artifact = wandb.Artifact(f"{output.stem}-teacher-labels", type="teacher-labels")
        artifact.add_file(str(output))
        run.log_artifact(artifact, aliases=["latest"])
        run.finish()
    print(f"[INFO] Saved Qwen affordance labels: {output}")


if __name__ == "__main__":
    main()
