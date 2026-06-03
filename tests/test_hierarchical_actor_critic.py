from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _obs(batch: int = 2):
    return {
        "camera_rgb": torch.rand(batch, 3, 32, 32),
        "map_crop": torch.ones(batch, 16, 16, dtype=torch.long),
        "frontier_mask": torch.zeros(batch, 16, 16),
        "trajectory_mask": torch.zeros(batch, 16, 16),
        "depth_line": torch.rand(batch, 64),
        "semantic_line": torch.zeros(batch, 64),
        "subgoal_features": torch.zeros(batch, 5),
    }


@unittest.skipUnless(torch is not None, "PyTorch is required")
class HierarchicalActorCriticTest(unittest.TestCase):
    def test_forward_and_action_evaluation(self):
        from exploration_stack.hierarchy import HierarchicalActorCritic, HierarchicalActorCriticConfig
        from exploration_stack.vlm_frontend import VLMPolicyEncoderConfig

        obs = _obs()
        obs["frontier_mask"][:, 7, 10] = 1.0
        model = HierarchicalActorCritic(
            HierarchicalActorCriticConfig(
                encoder=VLMPolicyEncoderConfig(
                    vlm_backend="mock",
                    image_mode="camera_plus_map",
                    image_size=32,
                    latent_dim=64,
                    depth_line_dim=64,
                    semantic_line_dim=64,
                    subgoal_feature_dim=5,
                ),
                max_candidates=8,
                candidate_hidden_dim=64,
                option_hidden_dim=128,
                option_embedding_dim=24,
            )
        )
        out = model(obs, training=False)
        self.assertEqual(tuple(out.action.shape), (2, 3))
        self.assertEqual(tuple(out.option_logits.shape), (2, 8))
        self.assertEqual(tuple(out.candidate_logits.shape), (2, 8))
        evaluated = model.evaluate_actions(obs, out.action, out.option, out.candidate)
        self.assertEqual(tuple(evaluated.log_prob.shape), (2,))
        self.assertTrue(torch.isfinite(evaluated.log_prob).all())


if __name__ == "__main__":
    unittest.main()
