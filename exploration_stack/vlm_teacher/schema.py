from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Direction = Literal["left", "center", "right"]
ViewDirection = Literal["left", "center", "right", "turn_around"]


@dataclass
class TeacherLabel:
    visible_navigation_affordances: dict[str, list[dict[str, Any]]]
    frontier_scores: list[dict[str, Any]]
    recommended_view_direction: ViewDirection
    uncertainty: float


def validate_teacher_label(payload: dict[str, Any]) -> TeacherLabel:
    required = {
        "visible_navigation_affordances",
        "frontier_scores",
        "recommended_view_direction",
        "uncertainty",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Teacher label missing keys: {sorted(missing)}")
    affordances = payload["visible_navigation_affordances"]
    if not isinstance(affordances, dict):
        raise ValueError("visible_navigation_affordances must be an object.")
    for key in ("doorway", "corridor", "open_space", "blocked"):
        if key not in affordances or not isinstance(affordances[key], list):
            raise ValueError(f"visible_navigation_affordances.{key} must be a list.")
    for item in affordances["doorway"]:
        bbox = item.get("bbox")
        confidence = item.get("confidence")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("doorway entries require bbox=[x0,y0,x1,y1].")
        _check_float(confidence, "doorway confidence", 0.0, 1.0)
    for key in ("corridor", "open_space", "blocked"):
        for item in affordances[key]:
            if item.get("direction") not in ("left", "center", "right"):
                raise ValueError(f"{key} direction must be left, center, or right.")
            _check_float(item.get("confidence"), f"{key} confidence", 0.0, 1.0)
    frontier_scores = payload["frontier_scores"]
    if not isinstance(frontier_scores, list):
        raise ValueError("frontier_scores must be a list.")
    for item in frontier_scores:
        if not isinstance(item.get("frontier_id"), int):
            raise ValueError("frontier_id must be an integer.")
        for key in ("score", "expected_information_gain", "risk", "doorway_likelihood"):
            _check_float(item.get(key), f"frontier_scores.{key}", 0.0, 1.0)
        if not isinstance(item.get("reason"), str):
            raise ValueError("frontier_scores.reason must be a string.")
    if payload["recommended_view_direction"] not in ("left", "center", "right", "turn_around"):
        raise ValueError("recommended_view_direction must be left, center, right, or turn_around.")
    _check_float(payload["uncertainty"], "uncertainty", 0.0, 1.0)
    return TeacherLabel(
        visible_navigation_affordances=affordances,
        frontier_scores=frontier_scores,
        recommended_view_direction=payload["recommended_view_direction"],
        uncertainty=float(payload["uncertainty"]),
    )


def _check_float(value, name: str, min_value: float, max_value: float) -> None:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")
    if not min_value <= float(value) <= max_value:
        raise ValueError(f"{name} must be in [{min_value}, {max_value}].")
