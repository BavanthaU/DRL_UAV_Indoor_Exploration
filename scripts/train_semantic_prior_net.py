#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exploration_stack.semantic.prior_net import SemanticPriorNet


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the small SemanticPriorNet from cached VLM/rule labels.")
    parser.add_argument("--labels_jsonl", default="data/semantic_priors/vlm_labels.jsonl")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    try:
        model = SemanticPriorNet()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    param_count = sum(param.numel() for param in model.parameters())
    print(f"SemanticPriorNet parameters: {param_count}")
    if args.dry_run:
        return 0
    labels_path = Path(args.labels_jsonl)
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} does not exist. Run scripts/label_semantic_priors_with_vlm.py first, "
            "or pass --dry_run to validate model construction only."
        )
    raise NotImplementedError(
        "Dataset tensorization/training loop is intentionally left for the next task. "
        "This script validates the small model and data entry point."
    )


if __name__ == "__main__":
    raise SystemExit(main())
