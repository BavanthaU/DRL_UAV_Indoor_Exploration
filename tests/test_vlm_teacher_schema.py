from __future__ import annotations

import unittest

from exploration_stack.vlm_teacher.schema import validate_teacher_label


class TeacherSchemaTest(unittest.TestCase):
    def test_valid_teacher_label(self):
        payload = {
            "visible_navigation_affordances": {
                "doorway": [{"bbox": [1, 2, 3, 4], "confidence": 0.7}],
                "corridor": [{"direction": "center", "confidence": 0.6}],
                "open_space": [{"direction": "left", "confidence": 0.5}],
                "blocked": [{"direction": "right", "confidence": 0.8}],
            },
            "frontier_scores": [
                {
                    "frontier_id": 1,
                    "score": 0.9,
                    "expected_information_gain": 0.8,
                    "risk": 0.2,
                    "doorway_likelihood": 0.4,
                    "reason": "visible opening",
                }
            ],
            "recommended_view_direction": "center",
            "uncertainty": 0.1,
        }
        label = validate_teacher_label(payload)
        self.assertEqual(label.recommended_view_direction, "center")

    def test_invalid_direction_rejected(self):
        payload = {
            "visible_navigation_affordances": {
                "doorway": [],
                "corridor": [{"direction": "up", "confidence": 0.5}],
                "open_space": [],
                "blocked": [],
            },
            "frontier_scores": [],
            "recommended_view_direction": "center",
            "uncertainty": 0.1,
        }
        with self.assertRaises(ValueError):
            validate_teacher_label(payload)


if __name__ == "__main__":
    unittest.main()
