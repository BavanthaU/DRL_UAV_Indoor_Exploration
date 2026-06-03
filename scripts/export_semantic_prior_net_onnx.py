#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exploration_stack.semantic.prior_net import SemanticPriorNet, torch


if torch is not None:

    class OnnxPriorWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, occupancy_crop, semantic_line, depth_line, frontier_features, room_graph_features):
            output = self.model(occupancy_crop, semantic_line, depth_line, frontier_features, room_graph_features)
            return torch.stack(
                [
                    output["frontier_score"],
                    output["doorway_likelihood"],
                    output["corridor_likelihood"],
                    output["risk"],
                    output["uncertainty"],
                ],
                dim=-1,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export SemanticPriorNet to ONNX for TensorRT/TorchScript workflows.")
    parser.add_argument("--output", default="outputs/semantic_prior_net.onnx")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("PyTorch is required for ONNX export.")
    model = SemanticPriorNet()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)
    model.eval()
    export_model = OnnxPriorWrapper(model).eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_inputs = (
        torch.zeros(1, 3, 32, 32),
        torch.zeros(1, 64),
        torch.zeros(1, 64),
        torch.zeros(1, 8),
        torch.zeros(1, 16),
    )
    torch.onnx.export(
        export_model,
        dummy_inputs,
        str(output_path),
        input_names=["occupancy_crop", "semantic_line", "depth_line", "frontier_features", "room_graph_features"],
        output_names=["semantic_prior"],
        opset_version=17,
    )
    print(f"exported {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
