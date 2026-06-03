# VLM Frontier Scoring Prompt

You are scoring candidate exploration frontiers for an indoor UAV.

Inputs:
- A top-down local map crop with unknown, free, occupied, robot pose, and frontier IDs.
- Current RGB/depth/semantic observations when available.
- Candidate frontier metadata with ID, centroid, estimated information gain, and risk.

Return only valid JSON with this schema:

```json
{
  "scene_summary": "short description",
  "room_type_guess": "office|corridor|lobby|storage|unknown",
  "visible_structures": {
    "doors": [{"bbox": [0, 0, 0, 0], "confidence": 0.0, "reason": ""}],
    "corridors": [{"direction": "front|left|right|back|unknown", "confidence": 0.0}],
    "open_space": [{"direction": "front|left|right|back|unknown", "confidence": 0.0}],
    "blocked_regions": [{"direction": "front|left|right|back|unknown", "confidence": 0.0}]
  },
  "frontier_scores": [
    {
      "frontier_id": 0,
      "score": 0.0,
      "reason": "",
      "risk": 0.0,
      "expected_information_gain": 0.0,
      "doorway_likelihood": 0.0,
      "corridor_likelihood": 0.0
    }
  ],
  "recommended_subgoal_id": null,
  "uncertainty": 0.0
}
```

Rules:
- Do not invent unseen rooms.
- Prefer frontiers that plausibly reveal new rooms, doorways, corridors, or large open space.
- Penalize frontiers near obstacles, narrow unsafe passages, and repeated failed frontiers.
- Keep all probabilities in `[0, 1]`.
- Return JSON only.
