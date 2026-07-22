"""Channel-mask logic and physical model rebuilding."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .models import PrunableConvBNReLU


def get_keep_masks(agents, threshold: float = 0.5):
    """Deterministic per-block keep mask (True = keep). Never empties a layer."""
    masks = []
    for agent in agents:
        with torch.no_grad():
            probs = agent()
        keep = probs >= threshold
        if keep.sum() == 0:  # safety: keep the single most-important channel
            keep = torch.zeros_like(probs, dtype=torch.bool)
            keep[probs.argmax()] = True
        masks.append(keep)
    return masks


def current_compression(agents, threshold: float = 0.5) -> float:
    """Percentage of channels currently below the keep threshold."""
    total = sum(a.weights.numel() for a in agents)
    dropped = 0
    for agent in agents:
        with torch.no_grad():
            dropped += int((agent() < threshold).sum().item())
    return 100.0 * dropped / total


def pruning_summary(model, agents, block_indices, threshold: float = 0.5):
    """Return (rows, total_channels, kept_channels). rows: (idx, keep, total)."""
    rows = []
    total = kept = 0
    for agent, idx in zip(agents, block_indices):
        n = model.features[idx].out_channels
        with torch.no_grad():
            k = int((agent() >= threshold).sum().item())
        rows.append((idx, k, n))
        total += n
        kept += k
    return rows, total, kept


def apply_masks(model, agents, block_indices, threshold: float = 0.5):
    """Set each block's channel_mask to the deterministic keep mask."""
    masks = get_keep_masks(agents, threshold)
    for m, idx in zip(masks, block_indices):
        model.features[idx].channel_mask = m.float().to(model.features[idx].channel_mask.device)


def build_pruned_model(model, agents, device, threshold: float = 0.5, avgpool_spatial: int = 1):
    """Rebuild the VGG with dropped channels physically removed.

    For each block: slice the conv (out + in channels) and its BN; the next
    block drops the corresponding input channels; the final Linear drops the
    flattened features of the removed last-block channels.
    """
    keep_masks = get_keep_masks(agents, threshold)
    pruned = copy.deepcopy(model).to(device)

    new_features = []
    prev_keep_idx = torch.arange(3, device=device)  # RGB input channels
    block_ptr = 0
    for layer in model.features:
        if isinstance(layer, PrunableConvBNReLU):
            conv, bn = layer.conv, layer.bn
            keep = keep_masks[block_ptr].to(device)
            keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)

            new_conv = nn.Conv2d(len(prev_keep_idx), len(keep_idx),
                                 kernel_size=conv.kernel_size, stride=conv.stride,
                                 padding=conv.padding, dilation=conv.dilation,
                                 groups=conv.groups, bias=conv.bias is not None)
            new_conv.weight.data.copy_(conv.weight.data[keep_idx][:, prev_keep_idx, :, :].clone())
            if conv.bias is not None:
                new_conv.bias.data.copy_(conv.bias.data[keep_idx].clone())

            new_bn = nn.BatchNorm2d(len(keep_idx))
            new_bn.weight.data.copy_(bn.weight.data[keep_idx].clone())
            new_bn.bias.data.copy_(bn.bias.data[keep_idx].clone())
            new_bn.running_mean.data.copy_(bn.running_mean.data[keep_idx].clone())
            new_bn.running_var.data.copy_(bn.running_var.data[keep_idx].clone())

            new_features.append(nn.Sequential(new_conv, new_bn, nn.ReLU(inplace=True)).to(device))
            prev_keep_idx = keep_idx
            block_ptr += 1
        else:
            new_features.append(copy.deepcopy(layer))
    pruned.features = nn.Sequential(*new_features).to(device)

    last_keep_idx = prev_keep_idx
    spatial = avgpool_spatial * avgpool_spatial
    old_fc = model.classifier[0]
    in_idx = (last_keep_idx.view(-1, 1) * spatial +
              torch.arange(spatial, device=device)).view(-1)
    new_fc = nn.Linear(len(last_keep_idx) * spatial, old_fc.out_features)
    new_fc.weight.data.copy_(old_fc.weight.data[:, in_idx].clone())
    new_fc.bias.data.copy_(old_fc.bias.data.clone())
    pruned.classifier = nn.Sequential(new_fc.to(device)).to(device)

    return pruned
