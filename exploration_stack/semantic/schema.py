from __future__ import annotations

from typing import Any

from exploration_stack.core import FrontierScore, SemanticPrior, VlmStructuredOutput

REQUIRED_TOP_LEVEL_KEYS = {
    "scene_summary",
    "room_type_guess",
    "visible_structures",
    "frontier_scores",
    "recommended_subgoal_id",
    "uncertainty",
}

VISIBLE_STRUCTURE_KEYS = {"doors", "corridors", "open_space", "blocked_regions"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_number(value: Any, name: str, *, min_value: float | None = None, max_value: float | None = None) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name} must be a number")
    number = float(value)
    if min_value is not None:
        _require(number >= min_value, f"{name} must be >= {min_value}")
    if max_value is not None:
        _require(number <= max_value, f"{name} must be <= {max_value}")
    return number


def validate_vlm_output(data: dict[str, Any]) -> VlmStructuredOutput:
    """Validate the strict VLM/LLM JSON output used for frontier scoring."""

    _require(isinstance(data, dict), "VLM output must be a JSON object")
    missing = REQUIRED_TOP_LEVEL_KEYS - set(data)
    _require(not missing, f"VLM output missing required keys: {sorted(missing)}")
    _require(isinstance(data["scene_summary"], str), "scene_summary must be a string")
    _require(isinstance(data["room_type_guess"], str), "room_type_guess must be a string")

    visible = data["visible_structures"]
    _require(isinstance(visible, dict), "visible_structures must be an object")
    missing_visible = VISIBLE_STRUCTURE_KEYS - set(visible)
    _require(not missing_visible, f"visible_structures missing keys: {sorted(missing_visible)}")
    for key in VISIBLE_STRUCTURE_KEYS:
        _require(isinstance(visible[key], list), f"visible_structures.{key} must be a list")

    frontier_scores = data["frontier_scores"]
    _require(isinstance(frontier_scores, list), "frontier_scores must be a list")
    parsed_scores: list[FrontierScore] = []
    for index, item in enumerate(frontier_scores):
        _require(isinstance(item, dict), f"frontier_scores[{index}] must be an object")
        for key in (
            "frontier_id",
            "score",
            "reason",
            "risk",
            "expected_information_gain",
            "doorway_likelihood",
            "corridor_likelihood",
        ):
            _require(key in item, f"frontier_scores[{index}] missing {key}")
        _require(isinstance(item["frontier_id"], int), f"frontier_scores[{index}].frontier_id must be int")
        _require(isinstance(item["reason"], str), f"frontier_scores[{index}].reason must be string")
        parsed_scores.append(
            FrontierScore(
                frontier_id=item["frontier_id"],
                score=_require_number(item["score"], f"frontier_scores[{index}].score", min_value=0.0, max_value=1.0),
                reason=item["reason"],
                risk=_require_number(item["risk"], f"frontier_scores[{index}].risk", min_value=0.0, max_value=1.0),
                expected_information_gain=_require_number(
                    item["expected_information_gain"],
                    f"frontier_scores[{index}].expected_information_gain",
                    min_value=0.0,
                ),
                doorway_likelihood=_require_number(
                    item["doorway_likelihood"],
                    f"frontier_scores[{index}].doorway_likelihood",
                    min_value=0.0,
                    max_value=1.0,
                ),
                corridor_likelihood=_require_number(
                    item["corridor_likelihood"],
                    f"frontier_scores[{index}].corridor_likelihood",
                    min_value=0.0,
                    max_value=1.0,
                ),
            )
        )

    recommended = data["recommended_subgoal_id"]
    _require(recommended is None or isinstance(recommended, int), "recommended_subgoal_id must be int or null")
    uncertainty = _require_number(data["uncertainty"], "uncertainty", min_value=0.0, max_value=1.0)
    return VlmStructuredOutput(
        scene_summary=data["scene_summary"],
        room_type_guess=data["room_type_guess"],
        visible_structures=visible,
        frontier_scores=parsed_scores,
        recommended_subgoal_id=recommended,
        uncertainty=uncertainty,
    )


def semantic_prior_from_vlm_output(output: VlmStructuredOutput) -> SemanticPrior:
    return SemanticPrior(
        scene_summary=output.scene_summary,
        room_type_guess=output.room_type_guess,
        frontier_scores=output.frontier_scores,
        recommended_subgoal_id=output.recommended_subgoal_id,
        uncertainty=output.uncertainty,
        planner_prior={"visible_structures": output.visible_structures},
    )
