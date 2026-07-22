"""Command-line entrypoints for the DECORE pipeline.

Examples:
    python -m decore.cli run --config configs/vgg16_cifar10.yaml
    python -m decore.cli run --config configs/vgg16_cifar10.yaml --smoke
    python -m decore.cli benchmark --config configs/vgg16_cifar10.yaml
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

from .config import Config
from .data import build_loaders
from .models import build_model
from .agents import build_agents
from .engine import run_baseline, run_decore, evaluate
from .pruning import build_pruned_model
from .metrics import count_parameters, count_flops, benchmark_latency
from .export import export_torchscript, export_onnx, check_parity
from .registry import log_run


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _log(msg: str):
    print(msg, flush=True)


def _load_agents_from_disk(agents, path, device):
    if os.path.exists(path):
        for a, s in zip(agents, torch.load(path, map_location=device)):
            a.load_state_dict(s)
        return True
    return False


def cmd_run(cfg: Config):
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = pick_device()
    _log(f"device: {device}")

    train_loader, test_loader = build_loaders(cfg.data_root, cfg.batch_size, cfg.num_workers)
    model = build_model(cfg.arch, cfg.num_classes, device)
    agents, block_indices = build_agents(model, device, cfg.init_weight)

    run_baseline(model, agents, block_indices, train_loader, test_loader, cfg, device, _log)
    best_acc, _ = run_decore(model, agents, block_indices, train_loader, test_loader, cfg, device, _log)

    # Build compressed model from best checkpoint.
    best_ckpt = os.path.join(cfg.out_dir, f"best_l{int(cfg.lambda_penalty)}.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device), strict=False)
    _load_agents_from_disk(agents, os.path.join(cfg.out_dir, f"agents_l{int(cfg.lambda_penalty)}.pth"), device)

    orig = {"params": count_parameters(model), "flops": count_flops(model)}
    pruned = build_pruned_model(model, agents, device, cfg.prune_threshold)
    p = {"params": count_parameters(pruned), "flops": count_flops(pruned)}

    parity = check_parity(model, pruned, test_loader, device)  # sanity: masked == pruned
    pruned_acc, _ = evaluate(pruned, test_loader, device, nn.CrossEntropyLoss())
    lat = benchmark_latency(pruned, device="cpu")
    base_lat = benchmark_latency(model, device="cpu")

    pruned_path = os.path.join(cfg.out_dir, f"pruned_l{int(cfg.lambda_penalty)}.pth")
    torch.save(pruned, pruned_path)

    params_pr = 100 * (1 - p["params"] / orig["params"])
    flops_pr = 100 * (1 - p["flops"] / orig["flops"])
    _log("=" * 60)
    _log(f"  DECORE-{int(cfg.lambda_penalty)} report ({cfg.arch}/{cfg.dataset})")
    _log("=" * 60)
    _log(f"  Params : {orig['params']/1e6:6.2f}M -> {p['params']/1e6:6.2f}M  ({params_pr:5.1f}% PR)")
    _log(f"  FLOPs  : {orig['flops']/1e6:6.2f}M -> {p['flops']/1e6:6.2f}M  ({flops_pr:5.1f}% PR)")
    _log(f"  CPU lat: {base_lat['median_ms']:6.2f}ms -> {lat['median_ms']:6.2f}ms")
    _log(f"  Top-1  : {pruned_acc:.2f}%   (masked==pruned diff={parity:.2e})")
    _log("=" * 60)

    log_run(cfg.out_dir, {
        "name": cfg.name, "lambda": cfg.lambda_penalty,
        "baseline_acc": best_acc, "pruned_acc": pruned_acc,
        "orig_params": orig["params"], "pruned_params": p["params"], "params_pr": params_pr,
        "orig_flops": orig["flops"], "pruned_flops": p["flops"], "flops_pr": flops_pr,
        "cpu_latency_ms": lat["median_ms"], "baseline_latency_ms": base_lat["median_ms"],
        "parity_diff": parity, "checkpoint": pruned_path,
    })
    _log(f"Saved compressed model -> {pruned_path}")


def cmd_benchmark(cfg: Config):
    device = pick_device()
    pruned_path = os.path.join(cfg.out_dir, f"pruned_l{int(cfg.lambda_penalty)}.pth")
    if not os.path.exists(pruned_path):
        raise FileNotFoundError(f"No compressed model at {pruned_path}; run `run` first.")
    model = torch.load(pruned_path, map_location="cpu", weights_only=False)
    lat = benchmark_latency(model, device="cpu")
    _log(f"CPU latency: median {lat['median_ms']:.2f}ms  "
         f"(min {lat['min_ms']:.2f}, max {lat['max_ms']:.2f}) | "
         f"params {count_parameters(model)/1e6:.2f}M flops {count_flops(model)/1e6:.2f}M")


def cmd_export(cfg: Config):
    pruned_path = os.path.join(cfg.out_dir, f"pruned_l{int(cfg.lambda_penalty)}.pth")
    model = torch.load(pruned_path, map_location="cpu", weights_only=False)
    ts = export_torchscript(model, os.path.join(cfg.out_dir, f"pruned_l{int(cfg.lambda_penalty)}.ts"))
    onnx = export_onnx(model, os.path.join(cfg.out_dir, f"pruned_l{int(cfg.lambda_penalty)}.onnx"))
    _log(f"Exported: {ts}  and  {onnx}")


def _apply_smoke(cfg: Config) -> Config:
    # Minimal schedule just to verify the pipeline runs end-to-end.
    return cfg.replace(baseline_epochs=1, num_epochs=2, policy_stop_epoch=1, print_every=1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="decore")
    ap.add_argument("command", choices=["run", "benchmark", "export"])
    ap.add_argument("--config", required=True, help="path to YAML config")
    ap.add_argument("--smoke", action="store_true", help="tiny schedule for a quick end-to-end test")
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    if args.smoke:
        cfg = _apply_smoke(cfg)

    {"run": cmd_run, "benchmark": cmd_benchmark, "export": cmd_export}[args.command](cfg)


if __name__ == "__main__":
    main()
