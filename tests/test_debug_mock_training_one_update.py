from __future__ import annotations

import unittest
from pathlib import Path


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required")
class DebugMockTrainingTest(unittest.TestCase):
    def test_one_update_train_path(self):
        from scripts._vlm_ppo_common import build_model, build_trainer, load_config, make_debug_env

        class Args:
            config = "configs/vlm_ppo_explorer/debug_mock_train.yaml"
            num_envs = 2
            max_iterations = 1
            seed = 11
            run_id = "unit_test"
            checkpoint = None
            ppo_backend = "torch_reference"
            device = "cpu"

        config = load_config(Path("configs/vlm_ppo_explorer/debug_mock_train.yaml"))
        env = make_debug_env(config, Args)
        model = build_model(config, action_dim=3)
        trainer, max_iterations = build_trainer(config, Args, model, Path("logs/vlm_ppo_explorer/unit_test"))
        result = trainer.train(env, max_iterations=max_iterations)
        self.assertGreater(result.timesteps, 0)
        self.assertEqual(result.updates, 1)
        self.assertIn("policy_loss", result.metrics)


if __name__ == "__main__":
    unittest.main()
