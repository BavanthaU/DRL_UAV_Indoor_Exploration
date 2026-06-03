from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required")
class OptionCreditAssignmentTest(unittest.TestCase):
    def test_reward_accumulates_and_resets_completed_options(self):
        from exploration_stack.hierarchy.credit_assignment import OptionCreditAccumulator, OptionCreditConfig

        credit = OptionCreditAccumulator(2, OptionCreditConfig(collision_penalty=2.0), device="cpu")
        reward = credit.update(
            coverage_gain=torch.tensor([0.2, 0.1]),
            new_cell_reward=torch.tensor([1.0, 0.0]),
            collision=torch.tensor([False, True]),
            stuck=torch.tensor([False, False]),
        )
        self.assertGreater(reward[0].item(), 0.0)
        self.assertLess(reward[1].item(), 0.0)
        returns, durations = credit.pop_completed(torch.tensor([True, False]))
        self.assertEqual(durations[0].item(), 1)
        self.assertEqual(credit.duration[0].item(), 0)
        self.assertEqual(credit.duration[1].item(), 1)
        self.assertGreater(returns[0].item(), 0.0)

    def test_option_manager_reset_forces_initial_resample(self):
        from exploration_stack.hierarchy.option_manager import OptionManager, OptionManagerConfig

        manager = OptionManager(2, OptionManagerConfig(option_interval_steps=4, max_option_age_steps=8), device="cpu")
        manager.reset()
        self.assertTrue(manager.should_resample(torch.zeros(2)).all())
        manager.update(torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([True, True]))
        self.assertFalse(manager.should_resample(torch.zeros(2)).any())


if __name__ == "__main__":
    unittest.main()
