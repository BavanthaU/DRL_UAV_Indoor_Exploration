from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import TeacherLabel, validate_teacher_label


@dataclass
class QwenTeacherConfig:
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    fallback_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_new_tokens: int = 512
    device_map: str = "auto"
    torch_dtype: str = "auto"


class Qwen25VLTeacher:
    """Offline Qwen2.5-VL labeler for saved exploration rollout frames."""

    def __init__(self, cfg: QwenTeacherConfig | None = None):
        self.cfg = cfg or QwenTeacherConfig()
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Qwen teacher requires transformers with Qwen2.5-VL support.") from exc
        self.processor = None
        self.model = None
        last_error: Exception | None = None
        for model_name in (self.cfg.model_name, self.cfg.fallback_model_name):
            try:
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=self.cfg.torch_dtype,
                    device_map=self.cfg.device_map,
                )
                self.model_name = model_name
                break
            except Exception as exc:  # pragma: no cover - depends on local model availability
                last_error = exc
        if self.processor is None or self.model is None:
            raise RuntimeError(
                "Could not load Qwen2.5-VL teacher. Download the 7B or 3B model, "
                "or pass a locally available model name."
            ) from last_error

    def label_image(self, image, prompt: str) -> TeacherLabel:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        output_ids = self.model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens)
        generated = self.processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
        payload = _extract_json(generated)
        return validate_teacher_label(payload)


def load_teacher_prompt(path: str | Path = "prompts/qwen_exploration_affordance_labeling.md") -> str:
    return Path(path).read_text(encoding="utf-8")


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))
