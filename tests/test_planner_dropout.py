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

    def test_candidate_dropout_restores_valid_candidate_when_first_is_invalid(self):
        from exploration_stack.planning.planner_dropout import PlannerDropoutConfig, PlannerFeatureDropout

        dropout = PlannerFeatureDropout(PlannerDropoutConfig(frontier_candidate_dropout_prob=1.0))
        features = torch.ones(3, 5, 18)
        mask = torch.tensor(
            [
                [False, False, True, True, False],
                [False, True, False, False, False],
                [True, False, False, False, False],
            ]
        )
        _, out_mask = dropout(features, mask, training=True)
        self.assertTrue(out_mask.any(dim=1).all())
        self.assertTrue(out_mask[0, 2])
        self.assertTrue(out_mask[1, 1])
        self.assertTrue(out_mask[2, 0])


if __name__ == "__main__":
    unittest.main()
