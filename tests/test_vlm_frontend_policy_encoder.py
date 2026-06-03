from __future__ import annotations

import unittest


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required")
class VlmPolicyEncoderTest(unittest.TestCase):
    def test_mock_encoder_outputs_policy_latents_and_aux(self):
        from exploration_stack.vlm_frontend import VLMPolicyEncoder, VLMPolicyEncoderConfig

        encoder = VLMPolicyEncoder(
            VLMPolicyEncoderConfig(
                vlm_backend="mock",
                image_mode="camera_plus_map",
                image_size=32,
                latent_dim=64,
                depth_line_dim=64,
                semantic_line_dim=64,
                subgoal_feature_dim=5,
            )
        )
        obs = {
            "camera_rgb": torch.rand(2, 3, 32, 32),
            "map_crop": torch.ones(2, 16, 16, dtype=torch.long),
            "frontier_mask": torch.zeros(2, 16, 16),
            "trajectory_mask": torch.zeros(2, 16, 16),
            "depth_line": torch.rand(2, 64),
            "semantic_line": torch.zeros(2, 64),
            "subgoal_features": torch.zeros(2, 5),
        }
        out = encoder(obs)
        self.assertEqual(tuple(out.z_actor.shape), (2, 64))
        self.assertIn("prompt_similarity", out.aux)
        self.assertEqual(out.aux["prompt_similarity"].shape[-1], 26)
        self.assertIn("map_embedding", out.aux)
        self.assertIn("frontier_utility", out.aux)
        self.assertIn("dead_end_likelihood", out.aux)


if __name__ == "__main__":
    unittest.main()
