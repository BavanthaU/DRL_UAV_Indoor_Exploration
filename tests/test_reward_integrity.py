from __future__ import annotations

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class RewardIntegrityTest(unittest.TestCase):
    def test_map_progress_training_config_excludes_privileged_rewards(self):
        config_text = (REPO_ROOT / "isaac45/RL_drone/env_config_training_map_progress.py").read_text(encoding="utf-8")
        forbidden = [
            "doorway_reward_per_drone",
            "fixed_area_covered_4_offices",
            "drone_covers_fixed_area_4_SCENES",
            "visited_doorways",
            "scene_limits",
            "room_reward",
        ]
        for token in forbidden:
            self.assertNotIn(token, config_text)

    def test_default_training_task_is_map_progress(self):
        train_text = (REPO_ROOT / "isaac45/train_exploration.py").read_text(encoding="utf-8")
        self.assertIn('default="Drone_SAC_no_IL_MapProgress_V1"', train_text)

    def test_map_progress_task_is_registered(self):
        registry_text = (REPO_ROOT / "isaac45/RL_drone/__init__.py").read_text(encoding="utf-8")
        self.assertIn('id="Drone_SAC_no_IL_MapProgress_V1"', registry_text)
        self.assertIn("env_config_training_map_progress.DroneMapProgressEnvCfg", registry_text)

    def test_new_stack_has_no_hard_coded_room_or_door_coordinates(self):
        forbidden = ["doorway_positions", "scene_limits", "visited_doorways"]
        for path in (REPO_ROOT / "exploration_stack").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{token} found in {path}")


if __name__ == "__main__":
    unittest.main()
