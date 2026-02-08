from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_BCE = nn.BCEWithLogitsLoss()
_CE = nn.CrossEntropyLoss()


def main_multi_positive_bce_loss(
    variant_emb: torch.Tensor,
    all_disease_emb: torch.Tensor,
    positive_disease_ids_per_variant: List[List[int]],
    temperature: float = 0.15,
    logit_scale: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Main task loss with logits [B, D] and multi-hot targets [B, D]."""
    logits = variant_emb @ all_disease_emb.t()
    if logit_scale is None:
        logits = logits / max(temperature, 1e-6)
    else:
        logits = logits * logit_scale
    targets = torch.zeros_like(logits)

    valid_rows = []
    for i, positives in enumerate(positive_disease_ids_per_variant):
        if not positives:
            continue
        idx = torch.tensor(sorted(set(positives)), device=logits.device, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < logits.shape[1])]
        if idx.numel() == 0:
            continue
        targets[i, idx] = 1.0
        valid_rows.append(i)

    if not valid_rows:
        return torch.zeros([], device=logits.device, dtype=logits.dtype)

    valid_rows_t = torch.tensor(valid_rows, device=logits.device, dtype=torch.long)
    per_elem = F.binary_cross_entropy_with_logits(
        logits.index_select(0, valid_rows_t),
        targets.index_select(0, valid_rows_t),
        reduction="none",
    )
    per_row = per_elem.mean(dim=1)
    if sample_weights is None:
        return per_row.mean()
    w = sample_weights.index_select(0, valid_rows_t).clamp_min(0.0)
    return (per_row * w).sum() / w.sum().clamp_min(1e-8)


def main_multi_positive_softmax_loss(
    variant_emb: torch.Tensor,
    all_disease_emb: torch.Tensor,
    positive_disease_ids_per_variant: List[List[int]],
    temperature: float = 0.15,
    logit_scale: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multi-positive retrieval loss with logits [B, D] over full disease bank."""
    logits = variant_emb @ all_disease_emb.t()
    if logit_scale is None:
        logits = logits / max(temperature, 1e-6)
    else:
        logits = logits * logit_scale
    pos_mask = torch.zeros_like(logits, dtype=torch.bool)

    for i, positives in enumerate(positive_disease_ids_per_variant):
        if not positives:
            continue
        idx = torch.tensor(sorted(set(positives)), device=logits.device, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < logits.shape[1])]
        if idx.numel() == 0:
            continue
        pos_mask[i, idx] = True

    valid = pos_mask.any(dim=1)
    if not valid.any():
        return torch.zeros([], device=logits.device, dtype=logits.dtype)

    den = torch.logsumexp(logits, dim=1)
    num = torch.logsumexp(logits.masked_fill(~pos_mask, float("-inf")), dim=1)
    per_row = -(num - den)
    per_row = per_row[valid]

    if sample_weights is None:
        return per_row.mean()
    w = sample_weights[valid].clamp_min(0.0)
    return (per_row * w).sum() / w.sum().clamp_min(1e-8)


def domain_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return _CE(logits, labels)


def mvp_reg_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    cosine = F.cosine_similarity(pred, target, dim=-1)
    return 1.0 - cosine.mean()


def func_impact_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    column_weights: Optional[torch.Tensor] = None,
    column_scales: Optional[torch.Tensor] = None,
    loss_type: str = "mse",
    smooth_l1_beta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    diff = pred - target
    if column_scales is not None:
        diff = diff / column_scales.clamp_min(1e-6)

    if loss_type == "smooth_l1":
        err = F.smooth_l1_loss(
            diff,
            torch.zeros_like(diff),
            reduction="none",
            beta=smooth_l1_beta,
        )
    else:
        err = diff**2

    if column_weights is not None:
        err = err * column_weights
        mask = mask * column_weights
    num = (err * mask).sum()
    den = mask.sum().clamp_min(eps)
    return num / den


def total_loss(
    losses: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    out = None
    for name, value in losses.items():
        if value is None:
            continue
        weighted = weights.get(name, 1.0) * value
        out = weighted if out is None else out + weighted
    if out is None:
        raise ValueError("No losses to aggregate")
    return out
