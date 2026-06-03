from __future__ import annotations

import json
from pathlib import Path
import unittest


class VlmPpoConfigTest(unittest.TestCase):
    def test_real_training_configs_use_real_vlm_backends(self):
        for name in ("rtx4080_mobileclip_train.yaml", "rtx4080_siglip_train.yaml", "jetson_export_student.yaml"):
            config = json.loads((Path("configs/vlm_ppo_explorer") / name).read_text(encoding="utf-8"))
            self.assertNotEqual(config["vlm"]["backend"], "mock", name)

    def test_debug_config_is_explicitly_debug_backend(self):
        config = json.loads(Path("configs/vlm_ppo_explorer/debug_mock_train.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["environment"]["backend"], "debug")
        self.assertEqual(config["vlm"]["backend"], "mock")
        self.assertFalse(config["wandb"]["enabled"])

    def test_real_configs_enable_wandb_uploads(self):
        for name in ("rtx4080_mobileclip_train.yaml", "rtx4080_siglip_train.yaml"):
            config = json.loads((Path("configs/vlm_ppo_explorer") / name).read_text(encoding="utf-8"))
            self.assertTrue(config["wandb"]["enabled"], name)
            self.assertTrue(config["wandb"]["log_checkpoints"], name)
            self.assertTrue(config["wandb"]["log_eval"], name)
            self.assertTrue(config["wandb"]["log_profile"], name)

    def test_cli_can_disable_wandb(self):
        from scripts._vlm_ppo_common import resolve_wandb_config

        class Args:
            wandb_mode = "disabled"
            wandb_project = None
            wandb_entity = None
            wandb_name = None
            wandb_group = None

        config = json.loads(Path("configs/vlm_ppo_explorer/rtx4080_mobileclip_train.yaml").read_text(encoding="utf-8"))
        resolved = resolve_wandb_config(config, Args, Path("logs/vlm_ppo_explorer/test"))
        self.assertFalse(resolved["enabled"])
        self.assertEqual(resolved["mode"], "disabled")


if __name__ == "__main__":
    unittest.main()
