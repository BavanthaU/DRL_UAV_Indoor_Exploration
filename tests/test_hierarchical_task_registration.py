from __future__ import annotations

import unittest


class HierarchicalTaskRegistrationTest(unittest.TestCase):
    def test_task_id_registers_without_isaac_import(self):
        import exploration_stack.tasks.vlm_hierarchical_ppo_exploration as task

        self.assertEqual(task.TASK_ID, "Isaac-VLM-Hierarchical-PPO-UAV-Exploration-v0")


if __name__ == "__main__":
    unittest.main()
