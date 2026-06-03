from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required")
class PlannerDropoutTest(unittest.TestCase):
    def test_planner_dropout_zeroes_features_and_preserves_first_candidate(self):
        from exploration_stack.planning.planner_dropout import PlannerDropoutConfig, PlannerFeatureDropout

        dropout = PlannerFeatureDropout(
            PlannerDropoutConfig(
                planner_dropout_prob=1.0,
                astar_feature_dropout_prob=1.0,
                frontier_candidate_dropout_prob=1.0,
            )
        )
        features = torch.ones(2, 4, 18)
        mask = torch.ones(2, 4, dtype=torch.bool)
        out_features, out_mask = dropout(features, mask, training=True)
        self.assertTrue(torch.allclose(out_features[:, :, 8:12], torch.zeros(2, 4, 4)))
        self.assertTrue(out_mask[:, 0].all())
        self.assertFalse(out_mask[:, 1:].any())

    def test_eval_mode_keeps_features(self):
        from exploration_stack.planning.planner_dropout import PlannerDropoutConfig, PlannerFeatureDropout

        dropout = PlannerFeatureDropout(PlannerDropoutConfig(planner_dropout_prob=1.0))
        features = torch.ones(1, 4, 18)
        mask = torch.ones(1, 4, dtype=torch.bool)
        out_features, out_mask = dropout(features, mask, training=False)
        self.assertTrue(torch.equal(features, out_features))
        self.assertTrue(torch.equal(mask, out_mask))


if __name__ == "__main__":
    unittest.main()
