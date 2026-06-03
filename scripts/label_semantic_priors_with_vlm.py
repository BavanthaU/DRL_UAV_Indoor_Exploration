#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exploration_stack.core import (
    DroneState,
    ExplorationMap,
    SensorPacket,
    SlamState,
    dataclass_from_dict,
    dataclass_to_dict,
)
from exploration_stack.semantic.vlm_reasoner import VlmSemanticReasoner


def mock_provider(packet: Any) -> dict[str, Any]:
    scores = []
    for frontier in packet.frontiers:
        scores.append(
            {
                "frontier_id": frontier.frontier_id,
                "score": min(1.0, frontier.information_gain / 50.0),
                "reason": "mock label for pipeline validation",
                "risk": 0.0,
                "expected_information_gain": frontier.information_gain,
                "doorway_likelihood": 0.0,
                "corridor_likelihood": 0.0,
            }
        )
    return {
        "scene_summary": "mock semantic label",
        "room_type_guess": "unknown",
        "visible_structures": {"doors": [], "corridors": [], "open_space": [], "blocked_regions": []},
        "frontier_scores": scores,
        "recommended_subgoal_id": scores[0]["frontier_id"] if scores else None,
        "uncertainty": 0.5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline VLM semantic-prior labeler for saved rollout JSONL.")
    parser.add_argument("--input_jsonl", required=True, help="Saved rollout samples with sensor/slam/map/frontier fields.")
    parser.add_argument("--output_jsonl", default="data/semantic_priors/vlm_labels.jsonl")
    parser.add_argument("--cache_dir", default="data/semantic_priors/cache")
    parser.add_argument("--mock", action="store_true", help="Use a deterministic mock provider instead of a VLM.")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reasoner = VlmSemanticReasoner(
        provider=mock_provider if args.mock else None,
        cache_dir=args.cache_dir,
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
        for line_index, line in enumerate(source):
            if not line.strip():
                continue
            sample = json.loads(line)
            sensor_packet = dataclass_from_dict(SensorPacket, sample.get("sensor_packet", {}))
            slam_state = dataclass_from_dict(SlamState, sample.get("slam_state", {}))
            exploration_map = dataclass_from_dict(ExplorationMap, sample.get("exploration_map", {}))
            robot_state = dataclass_from_dict(DroneState, sample.get("robot_state", {}))
            frontiers = exploration_map.frontiers
            prior = reasoner.infer(
                sensor_packet=sensor_packet,
                slam_state=slam_state,
                exploration_map=exploration_map,
                frontiers=frontiers,
                robot_state=robot_state,
            )
            sink.write(json.dumps({"sample_index": line_index, "semantic_prior": dataclass_to_dict(prior)}) + "\n")
    print(f"wrote semantic-prior labels to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
