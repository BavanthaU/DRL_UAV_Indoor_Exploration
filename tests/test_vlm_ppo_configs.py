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


if __name__ == "__main__":
    unittest.main()
