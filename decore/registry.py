"""Append-only run registry so results are queryable across experiments."""
from __future__ import annotations

import json
import os
import time
from typing import Optional


def log_run(out_dir: str, record: dict, path: Optional[str] = None) -> str:
    """Append a JSON record to runs.jsonl (one line per run)."""
    path = path or os.path.join(out_dir, "runs.jsonl")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return path


def load_runs(path: str):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def best_under_constraint(path: str, max_flops=None, max_params=None, min_acc=None):
    """Return the smallest-FLOP run meeting the given constraints (or None)."""
    runs = load_runs(path)
    def ok(r):
        if max_flops is not None and r.get("pruned_flops", 1e18) > max_flops:
            return False
        if max_params is not None and r.get("pruned_params", 1e18) > max_params:
            return False
        if min_acc is not None and r.get("pruned_acc", -1) < min_acc:
            return False
        return True
    cand = [r for r in runs if ok(r)]
    return min(cand, key=lambda r: r.get("pruned_flops", 1e18)) if cand else None
