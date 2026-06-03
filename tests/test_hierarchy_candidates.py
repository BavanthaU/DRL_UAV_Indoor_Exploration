from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required")
class HierarchyCandidateTest(unittest.TestCase):
    def test_candidate_builder_creates_frontier_and_special_candidates(self):
        from exploration_stack.hierarchy import CandidateBuilder, CandidateBuilderConfig

        obs = {"frontier_mask": torch.zeros(2, 16, 16), "camera_rgb": torch.zeros(2, 3, 8, 8)}
        obs["frontier_mask"][:, 8, 10] = 1.0
        builder = CandidateBuilder(CandidateBuilderConfig(max_candidates=8, feature_dim=18))
        features, mask = builder.build(obs)
        self.assertEqual(tuple(features.shape), (2, 8, 18))
        self.assertEqual(tuple(mask.shape), (2, 8))
        self.assertTrue(mask[:, 0].all())
        self.assertGreaterEqual(mask.sum(dim=1).min().item(), 4)
        self.assertTrue((features[:, 0, 0] == 1.0).all())

    def test_proposal_only_removes_planner_numeric_features(self):
        from exploration_stack.hierarchy import CandidateBuilder, CandidateBuilderConfig

        obs = {"frontier_mask": torch.zeros(1, 16, 16), "camera_rgb": torch.zeros(1, 3, 8, 8)}
        obs["frontier_mask"][0, 9, 12] = 1.0
        builder = CandidateBuilder(
            CandidateBuilderConfig(max_candidates=8, feature_dim=18, planner_feature_mode="proposal_only")
        )
        features, mask = builder.build(obs)
        self.assertTrue(mask[0, 0])
        self.assertTrue(torch.allclose(features[0, 0, 8:14], torch.zeros(6)))


if __name__ == "__main__":
    unittest.main()
