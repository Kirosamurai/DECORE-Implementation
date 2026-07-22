"""Configuration for a DECORE run.

A single dataclass captures every knob. It can be loaded from / dumped to
YAML so experiments are fully reproducible and a lambda-sweep is just a set
of YAML files.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Config:
    # --- experiment ---
    name: str = "vgg16_cifar10"
    seed: int = 42
    out_dir: str = "runs"          # where checkpoints/registry live
    data_root: str = "./data"

    # --- model / dataset ---
    arch: str = "vgg16"            # currently: vgg16 (CIFAR variant, with BN)
    dataset: str = "cifar10"
    num_classes: int = 10

    # --- schedule ---
    batch_size: int = 256
    baseline_epochs: int = 160     # pretrain full network before pruning
    num_epochs: int = 300          # DECORE joint-training epochs
    policy_stop_epoch: int = 260   # stop policy, then fine-tune the rest

    # --- optimisation ---
    baseline_lr: float = 0.1       # SGD LR for baseline pretraining (cosine)
    decore_lr: float = 0.1         # SGD LR during joint DECORE phase (keep net plastic)
    agent_lr: float = 0.05         # Adam LR for policy agents
    momentum: float = 0.9
    weight_decay: float = 5e-4

    # --- DECORE reward ---
    lambda_penalty: float = 50.0   # DECORE-<lambda>; lower => more compression
    prune_threshold: float = 0.5   # keep channel if sigmoid(weight) >= threshold
    init_weight: float = 6.9       # agent init => sigmoid(6.9) ~= 0.99

    # --- stabilisers (not in the paper; default off) ---
    use_reward_baseline: bool = False
    use_entropy_bonus: bool = False
    entropy_coef: float = 0.01
    baseline_momentum: float = 0.9

    # --- caching / logging ---
    retrain_baseline: bool = False  # reuse cached baseline checkpoint if present
    print_every: int = 5
    num_workers: int = 0            # 0 avoids macOS/Jupyter DataLoader pipe leaks

    def replace(self, **kwargs) -> "Config":
        return dataclasses.replace(self, **kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
