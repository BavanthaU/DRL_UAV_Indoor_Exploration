You are labeling a saved UAV exploration frame for auxiliary supervision.

Return only strict JSON with this schema:

{
  "visible_navigation_affordances": {
    "doorway": [{"bbox": [0, 0, 0, 0], "confidence": 0.0}],
    "corridor": [{"direction": "left|center|right", "confidence": 0.0}],
    "open_space": [{"direction": "left|center|right", "confidence": 0.0}],
    "blocked": [{"direction": "left|center|right", "confidence": 0.0}]
  },
  "frontier_scores": [
    {
      "frontier_id": 0,
      "score": 0.0,
      "expected_information_gain": 0.0,
      "risk": 0.0,
      "doorway_likelihood": 0.0,
      "reason": "short reason"
    }
  ],
  "recommended_view_direction": "left|center|right|turn_around",
  "uncertainty": 0.0
}

Use left, center, and right relative to the drone camera view. Set confidence,
score, risk, expected information gain, doorway likelihood, and uncertainty to
values in [0, 1]. Do not include explanations outside the JSON object.
