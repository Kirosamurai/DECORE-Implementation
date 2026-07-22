"""Training engine: baseline pretraining, DECORE joint training, fine-tuning."""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.optim as optim

from .pruning import current_compression, pruning_summary


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if criterion is not None:
            loss_sum += criterion(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total, (loss_sum / total if criterion is not None else 0.0)


def train_epoch(model, loader, optimizer, agents, agent_optimizers, block_indices,
                criterion, cfg, device, baselines=None):
    """One epoch. If agent_optimizers is None -> plain (baseline/finetune) training
    with the FIXED deterministic mask; else policy-learning with REINFORCE."""
    model.train()
    correct = total = 0
    loss_sum = 0.0

    for x, y in loader:
        bs = x.size(0)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        if agent_optimizers is not None:
            for opt in agent_optimizers:
                opt.zero_grad()

        log_probs_list, entropies, actions_list = [], [], []
        for agent, idx in zip(agents, block_indices):
            block = model.features[idx]
            probs = agent()
            if agent_optimizers is not None:
                m = torch.distributions.Bernoulli(probs)
                a = m.sample()
                log_probs_list.append(m.log_prob(a))
                entropies.append(m.entropy())
                actions_list.append(a)
                block.channel_mask = a.detach()
            else:
                block.channel_mask = (probs >= cfg.prune_threshold).float().detach()

        out = model(x)
        loss = criterion(out, y)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += bs
        loss_sum += loss.item() * bs

        loss.backward()
        optimizer.step()

        if agent_optimizers is not None:
            R_acc_mean = torch.where(
                pred == y, torch.ones(bs, device=device),
                -cfg.lambda_penalty * torch.ones(bs, device=device)).mean()
            for i, (opt, log_probs, ent, a) in enumerate(
                    zip(agent_optimizers, log_probs_list, entropies, actions_list)):
                R_iC = torch.sum(1 - a)
                R_i = R_iC * R_acc_mean
                if baselines is not None:
                    adv = R_i - baselines[i]
                    baselines[i] = (cfg.baseline_momentum * baselines[i]
                                    + (1 - cfg.baseline_momentum) * R_i.item())
                else:
                    adv = R_i
                policy_loss = -log_probs.sum() * adv
                if cfg.use_entropy_bonus:
                    policy_loss = policy_loss - cfg.entropy_coef * ent.sum()
                policy_loss.backward()
                opt.step()

    return loss_sum / total, 100.0 * correct / total


def run_baseline(model, agents, block_indices, train_loader, test_loader, cfg, device, log):
    ckpt = os.path.join(cfg.out_dir, "baseline.pth")
    criterion = nn.CrossEntropyLoss()
    if (not cfg.retrain_baseline) and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        acc, _ = evaluate(model, test_loader, device, criterion)
        log(f"Loaded cached baseline (skipped training). Top-1: {acc:.2f}%")
        return acc

    opt = optim.SGD(model.parameters(), lr=cfg.baseline_lr,
                    momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.baseline_epochs)
    best = 0.0
    for epoch in range(cfg.baseline_epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, agents, None,
                                      block_indices, criterion, cfg, device)
        te_acc, _ = evaluate(model, test_loader, device, criterion)
        sched.step()
        if te_acc > best:
            best = te_acc
            torch.save(model.state_dict(), ckpt)
        if (epoch + 1) % cfg.print_every == 0:
            log(f"[baseline] epoch {epoch+1:3d}/{cfg.baseline_epochs} "
                f"train {tr_acc:5.2f}% test {te_acc:5.2f}% best {best:5.2f}%")
    log(f"Baseline top-1: {best:.2f}%")
    return best


def run_decore(model, agents, block_indices, train_loader, test_loader, cfg, device, log):
    """Joint DECORE training + fine-tuning. Returns (best_acc, history)."""
    criterion = nn.CrossEntropyLoss()
    baseline_ckpt = os.path.join(cfg.out_dir, "baseline.pth")
    if os.path.exists(baseline_ckpt):
        model.load_state_dict(torch.load(baseline_ckpt, map_location=device), strict=False)
        log("Loaded pretrained baseline for DECORE phase.")

    opt = optim.SGD(model.parameters(), lr=cfg.decore_lr,
                    momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.num_epochs)
    agent_opts = [optim.Adam(a.parameters(), lr=cfg.agent_lr) for a in agents]
    baselines = [0.0 for _ in agents] if cfg.use_reward_baseline else None

    best_ckpt = os.path.join(cfg.out_dir, f"best_l{int(cfg.lambda_penalty)}.pth")
    agents_ckpt = os.path.join(cfg.out_dir, f"agents_l{int(cfg.lambda_penalty)}.pth")
    best = 0.0
    history = []

    for epoch in range(cfg.num_epochs):
        in_policy = epoch < cfg.policy_stop_epoch
        tr_loss, tr_acc = train_epoch(
            model, train_loader, opt, agents,
            agent_opts if in_policy else None,
            block_indices, criterion, cfg, device,
            baselines=(baselines if in_policy else None))
        te_acc, _ = evaluate(model, test_loader, device, criterion)
        comp = current_compression(agents, cfg.prune_threshold)
        sched.step()

        if te_acc > best:
            best = te_acc
            torch.save(model.state_dict(), best_ckpt)
            torch.save([a.state_dict() for a in agents], agents_ckpt)

        if (epoch + 1) % cfg.print_every == 0:
            phase = "policy" if in_policy else "finetune"
            log(f"[{phase:8s}] epoch {epoch+1:3d}/{cfg.num_epochs} "
                f"train {tr_acc:5.2f}% test {te_acc:5.2f}% compress {comp:5.1f}%")
        history.append({"epoch": epoch + 1, "train_acc": tr_acc,
                        "test_acc": te_acc, "compression": comp})

        if epoch + 1 == cfg.policy_stop_epoch:
            rows, total, kept = pruning_summary(model, agents, block_indices, cfg.prune_threshold)
            log(f"Policy done. Keeping {kept}/{total} channels "
                f"({100.0*(total-kept)/total:.1f}% pruned). Fine-tuning...")

    log(f"Best test accuracy: {best:.2f}%")
    return best, history
