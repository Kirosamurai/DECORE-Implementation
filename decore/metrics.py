"""Parameter / FLOP counting and CPU latency benchmarking."""
from __future__ import annotations

import time

import torch
import torch.nn as nn


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model, input_size=(1, 3, 32, 32)) -> int:
    """Multiply-accumulate operations (MACs) for one forward pass."""
    flops = [0]

    def conv_hook(m, i, o):
        oc, oh, ow = o.shape[1], o.shape[2], o.shape[3]
        flops[0] += oc * oh * ow * (m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups))

    def lin_hook(m, i, o):
        flops[0] += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        model(torch.randn(input_size).to(device))
    for h in hooks:
        h.remove()
    if was_training:
        model.train()
    return flops[0]


def benchmark_latency(model, input_size=(1, 3, 32, 32), device="cpu",
                      warmup: int = 10, iters: int = 50):
    """Median single-sample forward latency (ms) on the given device."""
    model = model.to(device).eval()
    x = torch.randn(input_size).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    median = times[len(times) // 2]
    return {"median_ms": median, "min_ms": times[0], "max_ms": times[-1]}


def model_report(model, input_size=(1, 3, 32, 32)) -> dict:
    return {"params": count_parameters(model), "flops": count_flops(model, input_size)}
