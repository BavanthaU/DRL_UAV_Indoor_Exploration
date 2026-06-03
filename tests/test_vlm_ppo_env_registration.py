from __future__ import annotations

import unittest

from exploration_stack.tasks.vlm_ppo_exploration.registration import TASK_ID, register_task


class VlmPpoRegistrationTest(unittest.TestCase):
    def test_task_id_is_stable(self):
        self.assertEqual(TASK_ID, "Isaac-VLM-PPO-UAV-Exploration-v0")

    def test_register_task_when_gym_available(self):
        try:
            import gymnasium as gym
        except ImportError:
            self.skipTest("gymnasium is not installed")
        register_task()
        self.assertIn(TASK_ID, gym.registry)


if __name__ == "__main__":
    unittest.main()
