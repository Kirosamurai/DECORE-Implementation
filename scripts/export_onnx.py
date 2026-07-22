"""Export the full and DECORE-pruned VGG-16 to ONNX for the in-browser demo.

Outputs:
    web/models/baseline.onnx   (full VGG-16)
    web/models/pruned.onnx     (DECORE-pruned)

Usage:
    python scripts/export_onnx.py
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from decore.models import build_model  # noqa: E402

OUT = os.path.join(ROOT, "web", "models")
BASELINE_CKPT = os.path.join(ROOT, "runs", "baseline.pth")
PRUNED_CKPT = os.path.join(ROOT, "runs", "pruned_vgg16_l50.pt")


def _export(model, path):
    model.eval()
    dummy = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18)
    # Consolidate into a single self-contained file (onnxruntime-web needs this).
    import onnx
    m = onnx.load(path)  # pulls in any external .data
    onnx.save_model(m, path, save_as_external_data=False)
    dfile = path + ".data"
    if os.path.exists(dfile):
        os.remove(dfile)
    print("wrote", path, f"({os.path.getsize(path)/1e6:.1f} MB)")


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cpu")

    baseline = build_model("vgg16", 10, dev)
    baseline.load_state_dict(torch.load(BASELINE_CKPT, map_location=dev), strict=False)
    _export(baseline, os.path.join(OUT, "baseline.onnx"))

    pruned = torch.load(PRUNED_CKPT, map_location=dev, weights_only=False)
    _export(pruned, os.path.join(OUT, "pruned.onnx"))


if __name__ == "__main__":
    main()
