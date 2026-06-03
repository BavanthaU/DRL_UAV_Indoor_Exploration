from __future__ import annotations

from dataclasses import dataclass

from .base import torch, F


@dataclass
class AuxiliaryLossOutput:
    loss: "torch.Tensor"
    terms: dict[str, float]


def compute_auxiliary_losses(predictions: dict, labels: dict | None) -> AuxiliaryLossOutput | None:
    """Compute optional VLM-head losses when offline labels are present."""

    if labels is None:
        return None
    losses = []
    terms: dict[str, float] = {}
    if "affordance_targets" in labels and "affordance_logits" in predictions:
        loss = F.binary_cross_entropy_with_logits(predictions["affordance_logits"], labels["affordance_targets"].float())
        losses.append(loss)
        terms["affordance_bce"] = float(loss.detach().cpu())
    if "frontier_score" in labels and "frontier_value" in predictions:
        loss = F.mse_loss(predictions["frontier_value"], labels["frontier_score"].float())
        losses.append(loss)
        terms["frontier_score_mse"] = float(loss.detach().cpu())
    if "collision_risk" in labels and "collision_risk" in predictions:
        loss = F.binary_cross_entropy(predictions["collision_risk"].clamp(1e-5, 1 - 1e-5), labels["collision_risk"].float())
        losses.append(loss)
        terms["collision_risk_bce"] = float(loss.detach().cpu())
    if "dead_end" in labels and "dead_end_likelihood" in predictions:
        loss = F.binary_cross_entropy(predictions["dead_end_likelihood"].clamp(1e-5, 1 - 1e-5), labels["dead_end"].float())
        losses.append(loss)
        terms["dead_end_bce"] = float(loss.detach().cpu())
    if "coverage_delta" in labels and "coverage_delta_prediction" in predictions:
        loss = F.mse_loss(predictions["coverage_delta_prediction"], labels["coverage_delta"].float())
        losses.append(loss)
        terms["coverage_delta_mse"] = float(loss.detach().cpu())
    if "option_label" in labels and "option_logits" in predictions:
        loss = F.cross_entropy(predictions["option_logits"], labels["option_label"].long())
        losses.append(loss)
        terms["option_prior_ce"] = float(loss.detach().cpu())
    if not losses:
        return None
    total = torch.stack(losses).sum()
    terms["aux_total"] = float(total.detach().cpu())
    return AuxiliaryLossOutput(loss=total, terms=terms)
