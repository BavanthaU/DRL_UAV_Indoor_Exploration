from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required")
class ExplorationMapExportTest(unittest.TestCase):
    def test_debug_env_map_snapshot_writes_png_and_metadata(self):
        from exploration_stack.evaluation import (
            BestExplorationMapTracker,
            capture_exploration_map,
            save_exploration_map_png,
            write_exploration_map_metadata,
        )
        from exploration_stack.tasks.vlm_ppo_exploration.debug_env import DebugVlmPpoEnvConfig, DebugVlmPpoVectorEnv

        env = DebugVlmPpoVectorEnv(DebugVlmPpoEnvConfig(num_envs=2, max_steps=8, device="cpu"))
        env.reset()
        snapshot = capture_exploration_map(env, 0)
        self.assertIsNotNone(snapshot)
        self.assertGreater(snapshot.mapped_free_cells, 0.0)
        self.assertEqual(snapshot.metadata()["source"], "agent_internal_map")

        tracker = BestExplorationMapTracker(env)
        tracker.record_completed(
            env_id=0,
            episode_index=1,
            episode_return=3.5,
            mapped_free_cells=snapshot.mapped_free_cells,
            episode_steps=4,
            global_step=4,
        )
        self.assertIsNotNone(tracker.best)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            image_path = save_exploration_map_png(
                tracker.best.snapshot,
                output_dir / "best_explored_map.png",
                metadata=tracker.best.metadata,
            )
            metadata_path = write_exploration_map_metadata(
                tracker.best.snapshot,
                output_dir / "best_explored_map.json",
                extra=tracker.best.metadata,
            )
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(image_path.exists())
            self.assertGreater(image_path.stat().st_size, 0)
            self.assertEqual(payload["selection"], "max_mapped_free_cells_then_return")
            self.assertNotIn("total_cells", payload)
            self.assertNotIn("coverage", payload)

    def test_training_map_logger_writes_live_map(self):
        from exploration_stack.evaluation import maybe_log_training_exploration_map
        from exploration_stack.tasks.vlm_ppo_exploration.debug_env import DebugVlmPpoEnvConfig, DebugVlmPpoVectorEnv

        class FakeLogger:
            def __init__(self, run_dir: Path):
                self.run_dir = run_dir
                self.images = []
                self.config = {
                    "log_train_maps": True,
                    "train_map_env_id": 1,
                    "train_map_image_size": 128,
                    "train_map_interval": 2,
                }

            def should_log_artifact(self, enabled_key=None):
                return enabled_key == "log_train_maps"

            def wandb_config_value(self, key, default=None):
                return self.config.get(key, default)

            def log_image(self, key, path, *, step=0, caption=None, enabled_key=None):
                self.images.append((key, Path(path), step, caption, enabled_key))

        env = DebugVlmPpoVectorEnv(DebugVlmPpoEnvConfig(num_envs=2, max_steps=8, device="cpu"))
        env.reset()
        with tempfile.TemporaryDirectory() as tmp:
            logger = FakeLogger(Path(tmp))
            skipped = maybe_log_training_exploration_map(
                logger,
                env,
                update=1,
                timesteps=8,
                metrics={"rollout/mean_reward": 0.25},
            )
            logged = maybe_log_training_exploration_map(
                logger,
                env,
                update=2,
                timesteps=16,
                metrics={"rollout/mean_reward": 0.5},
            )
            image_path = Path(tmp) / "train_maps" / "update_000002_env_001.png"
            metadata_path = Path(tmp) / "train_maps" / "update_000002_env_001.json"
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(skipped, {})
            self.assertTrue(image_path.exists())
            self.assertGreater(image_path.stat().st_size, 0)
            self.assertEqual(logger.images[0][0], "train/explored_map")
            self.assertEqual(logger.images[0][2], 2)
            self.assertGreater(logged["train_map/mapped_free_cells"], 0.0)
            self.assertEqual(payload["phase"], "train")
            self.assertEqual(payload["selection"], "configured_train_map_env")
            self.assertNotIn("total_cells", payload)
            self.assertNotIn("coverage", payload)


if __name__ == "__main__":
    unittest.main()
