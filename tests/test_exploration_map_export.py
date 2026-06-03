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


if __name__ == "__main__":
    unittest.main()
