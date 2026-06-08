from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _depth_obs(batch=4, steps=8, depth_dim=16, scalar_dim=8, action_dim=3, offset=0.0):
    depth = torch.rand(batch, steps, depth_dim) + offset
    scalars = torch.zeros(batch, steps, scalar_dim)
    scalars[..., 0] = depth.min(dim=-1).values
    scalars[..., 1] = depth.mean(dim=-1)
    scalars[..., 2] = depth.std(dim=-1)
    scalars[..., 3:6] = depth.mean(dim=-1, keepdim=True)
    return {
        "depth_rays": depth,
        "previous_action": torch.rand(batch, steps, action_dim) * 0.1,
        "depth_scalars": scalars,
    }


@unittest.skipUnless(torch is not None, "PyTorch is required")
class DepthHierarchicalRndBaselineTest(unittest.TestCase):
    def test_temporal_depth_rnd_target_frozen_and_predictor_updates(self):
        from exploration_stack.depth_hierarchical.curiosity import TemporalDepthRND, TemporalDepthRNDConfig

        torch.manual_seed(1)
        rnd = TemporalDepthRND(
            TemporalDepthRNDConfig(depth_dim=16, hidden_dim=48, feature_dim=32, rnd_lr=1.0e-3, normalize_reward=False)
        )
        obs = _depth_obs(depth_dim=16)
        target_before = [p.detach().clone() for p in rnd.target_encoder.parameters()]
        predictor_before = [p.detach().clone() for p in rnd.predictor_encoder.parameters()]
        loss = rnd.update_predictor(obs)
        self.assertGreater(loss, 0.0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(target_before, rnd.target_encoder.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(predictor_before, rnd.predictor_encoder.parameters())))

    def test_temporal_depth_rnd_novelty_and_repeated_training_decrease(self):
        from exploration_stack.depth_hierarchical.curiosity import TemporalDepthRND, TemporalDepthRNDConfig

        torch.manual_seed(2)
        rnd = TemporalDepthRND(
            TemporalDepthRNDConfig(depth_dim=16, hidden_dim=48, feature_dim=32, rnd_lr=2.0e-3, normalize_reward=False)
        )
        repeated = _depth_obs(batch=8, depth_dim=16)
        before = rnd(repeated, update_stats=False)[1].mean()
        for _ in range(40):
            rnd.update_predictor(repeated)
        after = rnd(repeated, update_stats=False)[1].mean()
        novel = _depth_obs(batch=8, depth_dim=16, offset=2.5)
        novel_error = rnd(novel, update_stats=False)[1].mean()
        self.assertLess(float(after), float(before))
        self.assertGreater(float(novel_error), float(after))

    def test_temporal_depth_rnd_zero_on_collision_and_near_obstacle(self):
        from exploration_stack.depth_hierarchical.curiosity import TemporalDepthRND, TemporalDepthRNDConfig

        rnd = TemporalDepthRND(TemporalDepthRNDConfig(depth_dim=16, hidden_dim=48, feature_dim=32))
        obs = _depth_obs(batch=2, depth_dim=16)
        reward, _ = rnd(obs, collision=torch.tensor([True, False]), min_depth=torch.tensor([1.0, 0.1]))
        self.assertEqual(float(reward[0]), 0.0)
        self.assertEqual(float(reward[1]), 0.0)

    def test_map_candidate_rnd_forward_shape(self):
        from exploration_stack.depth_hierarchical.curiosity import MapCandidateRND, MapCandidateRNDConfig

        rnd = MapCandidateRND(MapCandidateRNDConfig(feature_dim=32, hidden_dim=48))
        reward, raw = rnd(
            torch.rand(3, 4, 16, 16),
            torch.rand(3, 24),
            torch.rand(3, 8),
            torch.rand(3, 6),
        )
        self.assertEqual(tuple(reward.shape), (3,))
        self.assertEqual(tuple(raw.shape), (3,))

    def test_local_reward_has_no_new_cell_term_but_local_done_can_use_it(self):
        from exploration_stack.depth_hierarchical.local_explorer import LocalDoneEvaluator, compute_local_reward

        kwargs = dict(
            depth_rnd_reward=torch.ones(2),
            collision=torch.zeros(2),
            near_obstacle_penalty=torch.zeros(2),
            altitude_error=torch.zeros(2),
            action_smoothness=torch.zeros(2),
        )
        reward_a = compute_local_reward(**kwargs, new_cells_metric_only=torch.zeros(2))
        reward_b = compute_local_reward(**kwargs, new_cells_metric_only=torch.full((2,), 100.0))
        self.assertTrue(torch.allclose(reward_a, reward_b))
        done = LocalDoneEvaluator()(
            {
                "time_since_last_new_cell": torch.tensor([4.1, 0.0]),
                "local_new_cell_rate": torch.tensor([1.0, 1.0]),
                "max_open_depth": torch.tensor([2.0, 2.0]),
                "coverage_radius_fraction": torch.tensor([0.0, 0.0]),
                "repeated_cell_ratio": torch.tensor([0.0, 0.0]),
                "local_steps": torch.tensor([1, 1]),
            }
        )
        self.assertTrue(done.done[0])
        self.assertEqual(done.reason[0], "no_new_cells_for_s")

    def test_high_level_reward_uses_delayed_realized_gain(self):
        from exploration_stack.depth_hierarchical.high_level import compute_high_level_reward

        base = dict(
            map_rnd_reward=torch.zeros(1),
            missing_patch_filled=torch.zeros(1),
            new_connected_region_entered=torch.zeros(1),
            astar_path_length=torch.zeros(1),
            unreachable=torch.zeros(1),
            no_gain_after_reach=torch.zeros(1),
            collision_during_transit_or_local=torch.zeros(1),
            revisit_ratio_after_candidate=torch.zeros(1),
        )
        low, gain_low = compute_high_level_reward(new_free_cells=torch.zeros(1), unknown_to_known_cells=torch.zeros(1), **base)
        high, gain_high = compute_high_level_reward(
            new_free_cells=torch.full((1,), 500.0), unknown_to_known_cells=torch.zeros(1), **base
        )
        self.assertGreater(float(high), float(low))
        self.assertEqual(float(gain_high), 1.0)
        self.assertEqual(float(gain_low), 0.0)

    def test_candidate_selector_mask_and_astar_not_override(self):
        from exploration_stack.depth_hierarchical.high_level import HighLevelCandidateSelectorPPO
        from exploration_stack.depth_hierarchical.map_memory import Candidate, CandidateType, MapMemory
        from exploration_stack.depth_hierarchical.transit import AStarTransitNavigator

        selector = HighLevelCandidateSelectorPPO()
        features = torch.rand(1, 64, 24)
        mask = torch.zeros(1, 64, dtype=torch.bool)
        mask[:, :2] = True
        out = selector(features, mask, torch.rand(1, 4, 32, 32), torch.rand(1, 8), selected_index=torch.tensor([1]))
        self.assertEqual(float(out.probabilities[0, 2:].sum()), 0.0)
        self.assertTrue(torch.isfinite(out.log_prob).all())

        memory = MapMemory()
        memory.mark_observed(free_cells=[(r, c) for r in range(8, 20) for c in range(8, 20)], agent_cell=(10, 10))
        chosen = Candidate(42, CandidateType.MISSING_PATCH, 19, 19)
        plan = AStarTransitNavigator().plan_to_selected_candidate((10, 10), chosen, memory.grid)
        self.assertEqual(plan.selected_candidate_id, 42)
        self.assertEqual(plan.path[-1], (19, 19))

    def test_wandb_replay_logger_dry_run_and_mp4_renderer(self):
        from exploration_stack.depth_hierarchical.logging import WandbReplayLogger, WandbReplayLoggerConfig

        with tempfile.TemporaryDirectory() as tmp:
            logger = WandbReplayLogger(
                WandbReplayLoggerConfig(wandb_enabled=False, dry_run=True, output_dir=tmp, replay_resolution=128)
            )
            episode = {
                "map": [[-1, 0, 1], [0, 2, 0], [-1, 0, -1]],
                "trajectory": [(1, 1), (1, 2)],
                "depth_rays": [[0.5, 1.0, 2.0, 3.0]],
                "candidates": [{"row": 1, "col": 1}],
                "selected_candidate": 0,
                "scalars": {"coverage_percent": 10.0},
            }
            result = logger.log_replays(0, episode)
            self.assertIn("top_down_episode_replay", result)
            self.assertTrue((logger.output_dir / "top_down_episode_replay.mp4").exists())
            self.assertTrue((logger.output_dir / "depth_view_replay.mp4").exists())
            self.assertTrue((logger.output_dir / "map_candidate_replay.mp4").exists())

    def test_no_old_methods_enabled_in_baseline_config(self):
        from exploration_stack.depth_hierarchical.runtime import load_config

        cfg = load_config("configs/depth_hierarchical_ppo/train_joint_rnd_temporal_wandb.yaml")
        baseline = cfg["baseline"]
        self.assertFalse(baseline["use_vlm"])
        self.assertFalse(baseline["use_convnext"])
        self.assertFalse(baseline["use_il"])
        self.assertFalse(baseline["use_sac"])
        self.assertFalse(baseline["astar_selects_candidate"])
        self.assertEqual(baseline["curiosity_type"], "rnd")
        self.assertTrue(baseline["learned_high_level_selector"])


if __name__ == "__main__":
    unittest.main()
