from __future__ import annotations

import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required")
class HierarchicalTrainingTest(unittest.TestCase):
    def test_one_update_train_path(self):
        from scripts._vlm_hierarchical_common import (
            build_hierarchical_model,
            build_hierarchical_trainer,
            load_config,
            make_env,
        )

        class Args:
            config = "configs/vlm_hierarchical_ppo/debug_mock.yaml"
            task = "Isaac-VLM-Hierarchical-PPO-UAV-Exploration-v0"
            num_envs = 2
            max_iterations = 1
            seed = 13
            run_id = "unit_test_hierarchical"
            checkpoint = None
            ppo_backend = "torch_reference"
            device = "cpu"
            wandb_mode = "disabled"
            wandb_project = None
            wandb_entity = None
            wandb_name = None
            wandb_group = None

        config = load_config(Path("configs/vlm_hierarchical_ppo/debug_mock.yaml"))
        env = make_env(config, Args)
        model = build_hierarchical_model(config, action_dim=3)
        trainer, max_iterations = build_hierarchical_trainer(
            config,
            Args,
            model,
            Path("logs/vlm_hierarchical_ppo/unit_test"),
        )
        result = trainer.train(env, max_iterations=max_iterations)
        self.assertGreater(result.timesteps, 0)
        self.assertEqual(result.updates, 1)
        self.assertIn("policy_loss", result.metrics)
        self.assertIn("option_fraction/ROTATE_SCAN", result.metrics)


if __name__ == "__main__":
    unittest.main()
