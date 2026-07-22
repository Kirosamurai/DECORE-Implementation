"""Export a compressed model to TorchScript / ONNX with an accuracy-parity check."""
from __future__ import annotations

import torch

from .engine import evaluate


def export_torchscript(model, path: str, input_size=(1, 3, 32, 32), device="cpu"):
    model = model.to(device).eval()
    example = torch.randn(input_size).to(device)
    scripted = torch.jit.trace(model, example)
    scripted.save(path)
    return path


def export_onnx(model, path: str, input_size=(1, 3, 32, 32), device="cpu", opset: int = 17):
    model = model.to(device).eval()
    example = torch.randn(input_size).to(device)
    torch.onnx.export(
        model, example, path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset)
    return path


@torch.no_grad()
def check_parity(model_a, model_b, loader, device, tol: float = 1e-3):
    """Confirm two models agree (max logit diff) on a batch; returns max abs diff."""
    model_a.eval(); model_b.eval()
    x, _ = next(iter(loader))
    x = x.to(device)
    diff = (model_a(x) - model_b(x)).abs().max().item()
    return diff
