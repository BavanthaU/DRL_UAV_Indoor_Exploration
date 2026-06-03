from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required")
class HierarchyPolicyTest(unittest.TestCase):
    def test_option_mask_falls_back_to_rotate_scan_when_no_candidates(self):
        from exploration_stack.hierarchy import NUM_OPTIONS, build_option_mask
        from exploration_stack.hierarchy.option_types import ExplorationOption

        features = torch.zeros(2, 4, 18)
        candidate_mask = torch.zeros(2, 4, dtype=torch.bool)
        option_mask = build_option_mask(features, candidate_mask)
        self.assertEqual(tuple(option_mask.shape), (2, NUM_OPTIONS))
        self.assertTrue(option_mask[:, int(ExplorationOption.ROTATE_SCAN)].all())
        self.assertEqual(option_mask.sum().item(), 2)

    def test_option_policy_forward_shapes_and_candidate_masking(self):
        from exploration_stack.hierarchy import SemanticOptionPolicy
        from exploration_stack.hierarchy.option_policy import OptionPolicyConfig

        policy = SemanticOptionPolicy(OptionPolicyConfig(context_dim=64, candidate_dim=32, hidden_dim=64))
        context = torch.randn(3, 64)
        tokens = torch.randn(3, 5, 32)
        pooled = torch.randn(3, 32)
        mask = torch.tensor([[True, True, False, False, False], [True, False, False, False, False], [True, True, True, True, True]])
        option_logits, candidate_logits, termination_prob, confidence = policy(context, tokens, pooled, mask)
        self.assertEqual(tuple(option_logits.shape), (3, 8))
        self.assertEqual(tuple(candidate_logits.shape), (3, 5))
        self.assertLess(candidate_logits[0, 2].item(), -1.0e8)
        self.assertEqual(tuple(termination_prob.shape), (3,))
        self.assertEqual(tuple(confidence.shape), (3,))

    def test_depth_line_safety_shield_logs_intervention_condition(self):
        from exploration_stack.hierarchy import DepthLineSafetyShield
        from exploration_stack.hierarchy.safety_shield import DepthLineSafetyShieldConfig

        shield = DepthLineSafetyShield(DepthLineSafetyShieldConfig(enabled=True, min_depth_fraction=0.1))
        actions = torch.tensor([[0.8, 0.0, 0.0], [-0.2, 0.0, 0.0]])
        obs = {"depth_line": torch.ones(2, 9)}
        obs["depth_line"][0, 4] = 0.05
        shielded, interventions = shield(actions, obs)
        self.assertEqual(shielded[0, 0].item(), 0.0)
        self.assertTrue(interventions[0])
        self.assertFalse(interventions[1])


if __name__ == "__main__":
    unittest.main()
