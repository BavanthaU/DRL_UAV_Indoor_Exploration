from __future__ import annotations

import unittest


class MobileClipBackendTest(unittest.TestCase):
    def test_model_name_aliases_are_normalized(self):
        from exploration_stack.vlm_frontend.mobileclip_encoder import MobileCLIPEncoder

        self.assertEqual(MobileCLIPEncoder._normalize_model_name("mobileclip_s2"), "MobileCLIP-S2")
        self.assertEqual(MobileCLIPEncoder._normalize_model_name("mobileclip-s0"), "MobileCLIP-S0")
        self.assertEqual(MobileCLIPEncoder._normalize_model_name("MobileCLIP-S1"), "MobileCLIP-S1")


if __name__ == "__main__":
    unittest.main()
