from __future__ import annotations

import json
from pathlib import Path
import unittest


class HierarchicalConfigTest(unittest.TestCase):
    def test_debug_config_is_mock_and_no_privileged_map(self):
        config = json.loads(Path("configs/vlm_hierarchical_ppo/debug_mock.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["environment"]["backend"], "debug")
        self.assertEqual(config["vlm"]["backend"], "mock")
        self.assertFalse(config["hierarchy"]["use_privileged_map_for_training"])
        self.assertNotIn("success_threshold", config["environment"])
        self.assertIn("frontier_closed_steps", config["environment"])
        self.assertIn("return_home_fraction", config["environment"])
        self.assertFalse(config["wandb"]["enabled"])

    def test_main_config_uses_features_only_with_dropout(self):
        config = json.loads(Path("configs/vlm_hierarchical_ppo/planner_dropout_main.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["hierarchy"]["variant"], "B7_main")
        self.assertEqual(config["hierarchy"]["planner_feature_mode"], "features_only")
        self.assertEqual(config["hierarchy"]["planner_dropout_prob"], 0.2)
        self.assertEqual(config["hierarchy"]["astar_feature_dropout_prob"], 0.2)
        self.assertEqual(config["hierarchy"]["frontier_candidate_dropout_prob"], 0.1)
        self.assertNotIn("success_threshold", config["environment"]["map"])
        self.assertEqual(config["environment"]["map"]["return_home_after_s"], 480.0)
        self.assertEqual(config["environment"]["action"]["min_altitude_m"], 0.6)
        self.assertEqual(config["environment"]["action"]["max_altitude_m"], 2.2)
        self.assertEqual(config["environment"]["action"]["altitude_violation_steps"], 4)
        self.assertFalse(config["hierarchy"]["use_privileged_map_for_training"])
        self.assertNotEqual(config["vlm"]["backend"], "mock")
        self.assertTrue(config["wandb"]["enabled"])

    def test_required_baseline_configs_exist(self):
        required = {
            "rtx4080_mobileclip_main.yaml",
            "rtx4080_siglip_ablation.yaml",
            "classical_frontier_baseline.yaml",
            "flat_vlm_ppo_baseline.yaml",
            "no_vlm_hierarchical_baseline.yaml",
            "no_astar_features_ablation.yaml",
            "qwen_auxiliary_ablation.yaml",
            "planner_dropout_main.yaml",
            "jetson_deploy_student.yaml",
        }
        existing = {path.name for path in Path("configs/vlm_hierarchical_ppo").glob("*.yaml")}
        self.assertTrue(required.issubset(existing))


if __name__ == "__main__":
    unittest.main()
