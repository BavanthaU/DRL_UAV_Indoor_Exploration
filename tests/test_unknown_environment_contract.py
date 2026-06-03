from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required")
class UnknownEnvironmentContractTest(unittest.TestCase):
    def test_debug_env_logs_map_progress_without_total_area_ratio(self):
        from exploration_stack.tasks.vlm_ppo_exploration.debug_env import DebugVlmPpoEnvConfig, DebugVlmPpoVectorEnv

        env = DebugVlmPpoVectorEnv(DebugVlmPpoEnvConfig(num_envs=2, max_steps=8, device="cpu"))
        obs = env.reset()
        next_obs, reward, done, info = env.step(torch.zeros(2, 3))
        terms = info["reward_terms"]
        self.assertIn("mapped_free_cells", terms)
        self.assertIn("frontier_count", terms)
        self.assertIn("frontier_closed", terms)
        self.assertIn("return_home_phase", terms)
        self.assertNotIn("coverage", terms)
        self.assertNotIn("success", terms)
        self.assertEqual(tuple(next_obs["map_crop"].shape), (2, 16, 16))
        self.assertEqual(tuple(reward.shape), (2,))
        self.assertEqual(tuple(done.shape), (2,))

    def test_reward_terms_use_frontier_closure_not_area_threshold(self):
        from exploration_stack.tasks.vlm_ppo_exploration.reward_terms import RewardWeights, compute_extrinsic_reward

        reward, terms = compute_extrinsic_reward(
            map_progress=torch.tensor([0.5]),
            frontier_closed=torch.tensor([True]),
            collision=torch.tensor([False]),
            esdf_clearance=torch.tensor([1.0]),
            safety_radius=0.45,
            dt=1.0,
            idle_mask=torch.tensor([False]),
            yaw_flip=torch.tensor([False]),
            action=torch.zeros(1, 3),
            prev_action=torch.zeros(1, 3),
            altitude=torch.tensor([1.2]),
            target_altitude=1.2,
            weights=RewardWeights(),
        )
        self.assertGreater(reward.item(), 0.0)
        self.assertIn("frontier_closure", terms)
        self.assertIn("map_progress", terms)
        self.assertNotIn("success", terms)
        self.assertNotIn("global_coverage_progress", terms)

    def test_reward_terms_sanitize_nonfinite_pose_inputs(self):
        from exploration_stack.tasks.vlm_ppo_exploration.reward_terms import RewardWeights, combine_rewards, compute_extrinsic_reward

        extrinsic, extrinsic_terms = compute_extrinsic_reward(
            map_progress=torch.tensor([1.0, float("nan")]),
            frontier_closed=torch.tensor([False, False]),
            collision=torch.tensor([False, True]),
            esdf_clearance=torch.tensor([1.0, float("nan")]),
            safety_radius=0.45,
            dt=1.0,
            idle_mask=torch.tensor([False, False]),
            yaw_flip=torch.tensor([False, False]),
            action=torch.tensor([[0.0, 0.0, 0.0], [float("inf"), 0.0, 0.0]]),
            prev_action=torch.zeros(2, 3),
            altitude=torch.tensor([float("nan"), float("inf")]),
            target_altitude=1.2,
            return_home_progress=torch.tensor([float("nan"), 1.0]),
            weights=RewardWeights(),
        )
        reward, intrinsic_terms = combine_rewards(
            extrinsic,
            torch.tensor([1.0, float("nan")]),
            torch.zeros(2),
            torch.zeros(2),
            RewardWeights(),
        )
        self.assertTrue(torch.isfinite(extrinsic).all())
        self.assertTrue(torch.isfinite(reward).all())
        for value in {**extrinsic_terms, **intrinsic_terms}.values():
            self.assertTrue(torch.isfinite(value).all())

    def test_nonfinite_altitude_is_terminal(self):
        from exploration_stack.tasks.vlm_ppo_exploration.terminations import altitude_out_of_bounds

        done = altitude_out_of_bounds(torch.tensor([1.2, float("nan"), float("inf")]), 0.8, 1.8)
        self.assertEqual(done.tolist(), [False, True, True])


if __name__ == "__main__":
    unittest.main()
