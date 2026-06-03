from __future__ import annotations

from dataclasses import dataclass

from .base import torch


@dataclass
class AffordanceSummary:
    doorway_likelihood: "torch.Tensor"
    corridor_likelihood: "torch.Tensor"
    open_space_likelihood: "torch.Tensor"
    collision_risk: "torch.Tensor"
    frontier_value: "torch.Tensor"


def affordance_summary_from_prompt_similarity(prompt_similarity, prompt_bank) -> AffordanceSummary:
    probs = torch.softmax(prompt_similarity, dim=-1)
    return AffordanceSummary(
        doorway_likelihood=probs[:, prompt_bank.doorway_index],
        corridor_likelihood=probs[:, prompt_bank.corridor_index],
        open_space_likelihood=probs[:, prompt_bank.open_space_index],
        collision_risk=torch.maximum(probs[:, prompt_bank.obstacle_index], probs[:, prompt_bank.risky_index]),
        frontier_value=probs[:, prompt_bank.frontier_index],
    )


def patch_affordance_map(patch_tokens, text_embeddings):
    if patch_tokens is None:
        return None
    patch_tokens = torch.nn.functional.normalize(patch_tokens, dim=-1)
    text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
    return patch_tokens @ text_embeddings.T

