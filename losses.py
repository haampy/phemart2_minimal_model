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
) -> torch.Tensor:
    """Main task loss with logits [B, D] and multi-hot targets [B, D]."""
    logits = variant_emb @ all_disease_emb.t() / max(temperature, 1e-6)
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
    return _BCE(logits.index_select(0, valid_rows_t), targets.index_select(0, valid_rows_t))


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
    eps: float = 1e-8,
) -> torch.Tensor:
    err = (pred - target) ** 2
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
