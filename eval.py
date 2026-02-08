from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from model import MultiTaskModel


def _safe_binary_auc(scores: torch.Tensor, labels01: torch.Tensor) -> float | None:
    labels01 = labels01.to(torch.float32)
    n_pos = int(labels01.sum().item())
    n_all = int(labels01.numel())
    n_neg = n_all - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    ranks = torch.argsort(torch.argsort(scores, dim=0), dim=0).to(torch.float32) + 1.0
    sum_rank_pos = ranks[labels01 > 0.5].sum()
    auc = (sum_rank_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc.item())


def _safe_average_precision(scores: torch.Tensor, labels01: torch.Tensor) -> float | None:
    labels01 = labels01.to(torch.float32)
    n_pos = float(labels01.sum().item())
    if n_pos <= 0:
        return None

    order = torch.argsort(scores, descending=True)
    y = labels01.index_select(0, order)
    tp_cum = torch.cumsum(y, dim=0)
    k = torch.arange(1, y.numel() + 1, device=y.device, dtype=torch.float32)
    precision_at_k = tp_cum / k
    ap = (precision_at_k * y).sum() / n_pos
    return float(ap.item())


def _safe_pearson(x: torch.Tensor, y: torch.Tensor) -> float | None:
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    if x.numel() == 0 or y.numel() == 0 or x.numel() != y.numel():
        return None
    x = x - x.mean()
    y = y - y.mean()
    den = torch.sqrt((x * x).sum() * (y * y).sum()).item()
    if den <= 1e-12:
        return None
    return float(((x * y).sum().item()) / den)


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    return torch.argsort(torch.argsort(x, dim=0), dim=0).to(torch.float32) + 1.0


def _safe_spearman(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() == 0 or y.numel() == 0 or x.numel() != y.numel():
        return None
    rx = _rankdata(x.to(torch.float32))
    ry = _rankdata(y.to(torch.float32))
    return _safe_pearson(rx, ry)


def _mean_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def evaluate_main(
    model: MultiTaskModel,
    loader,
    variant_x: torch.Tensor,
    protein_x: torch.Tensor,
    gene_graph_emb: torch.Tensor,
    trait_graph_emb: torch.Tensor,
    disease_ids: Sequence[int],
    disease_to_traits: Dict[int, List[int]],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    disease_ids = list(disease_ids)
    disease_id_to_col = {d: i for i, d in enumerate(disease_ids)}

    with torch.no_grad():
        disease_emb = model.encode_disease_batch(disease_ids, disease_to_traits, trait_graph_emb)

    reciprocal_ranks: List[float] = []
    recall_counts = {1: 0, 5: 0, 10: 0}
    query_aurocs: List[float] = []
    query_auprcs: List[float] = []
    n = 0

    with torch.no_grad():
        for batch in loader:
            variant_idx = batch["variant_idx"].to(device)
            gene_idx = batch["gene_idx"].to(device)
            z_v = model.encode_variant(
                variant_idx,
                gene_idx,
                variant_x,
                protein_x,
                gene_graph_emb,
            )
            z_v = F.normalize(model.clip_variant_proj(z_v), dim=-1)
            scores = z_v @ disease_emb.t()  # [B, D]

            for i, positives in enumerate(batch["positive_disease_ids"]):
                cols = [disease_id_to_col[d] for d in positives if d in disease_id_to_col]
                if not cols:
                    continue
                ranking = torch.argsort(scores[i], descending=True)
                rank_map = torch.empty_like(ranking)
                rank_map[ranking] = torch.arange(1, ranking.numel() + 1, device=ranking.device)
                best_rank = int(rank_map[torch.tensor(cols, device=ranking.device)].min().item())

                reciprocal_ranks.append(1.0 / best_rank)
                for k in recall_counts:
                    if best_rank <= k:
                        recall_counts[k] += 1

                labels01 = torch.zeros(scores.shape[1], dtype=torch.float32, device=scores.device)
                labels01[torch.tensor(cols, device=scores.device)] = 1.0
                row_scores = scores[i]
                auc = _safe_binary_auc(row_scores, labels01)
                ap = _safe_average_precision(row_scores, labels01)
                if auc is not None:
                    query_aurocs.append(auc)
                if ap is not None:
                    query_auprcs.append(ap)
                n += 1

    if n == 0:
        return {
            "mrr": 0.0,
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "auroc_query_mean": 0.0,
            "auprc_query_mean": 0.0,
            "map": 0.0,
            "n_eval": 0.0,
        }

    return {
        "mrr": float(sum(reciprocal_ranks) / n),
        "recall@1": float(recall_counts[1] / n),
        "recall@5": float(recall_counts[5] / n),
        "recall@10": float(recall_counts[10] / n),
        "auroc_query_mean": _mean_or_zero(query_aurocs),
        "auprc_query_mean": _mean_or_zero(query_auprcs),
        "map": _mean_or_zero(query_auprcs),
        "n_eval": float(n),
    }


def evaluate_domain(
    model: MultiTaskModel,
    loader,
    variant_x: torch.Tensor,
    protein_x: torch.Tensor,
    gene_graph_emb: torch.Tensor,
    domain_embeddings: torch.Tensor,
    device: torch.device,
    temperature: float = 0.15,
) -> Dict[str, float]:
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    n = 0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            variant_idx = batch["variant_idx"].to(device)
            gene_idx = batch["gene_idx"].to(device)
            labels = batch["label"].to(device)
            logits = model.forward_domain(
                variant_idx,
                gene_idx,
                variant_x,
                protein_x,
                gene_graph_emb,
                domain_embeddings,
                temperature=temperature,
            )

            top1 = logits.argmax(dim=-1)
            top5 = logits.topk(k=min(5, logits.shape[1]), dim=-1).indices
            correct_top1 += int((top1 == labels).sum().item())
            correct_top5 += int((top5 == labels.unsqueeze(1)).any(dim=1).sum().item())
            n += labels.shape[0]
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if n == 0:
        return {
            "top1": 0.0,
            "top5": 0.0,
            "macro_f1": 0.0,
            "balanced_acc": 0.0,
            "ovr_macro_auroc": 0.0,
            "ovr_macro_auprc": 0.0,
            "n_eval": 0.0,
        }

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    preds = logits_cat.argmax(dim=-1)
    active_classes = torch.unique(labels_cat).tolist()

    recalls: List[float] = []
    f1s: List[float] = []
    aucs: List[float] = []
    aps: List[float] = []
    for c in active_classes:
        c = int(c)
        y_true = (labels_cat == c).to(torch.float32)
        support = float(y_true.sum().item())
        if support <= 0:
            continue

        y_pred = (preds == c).to(torch.float32)
        tp = float((y_true * y_pred).sum().item())
        fp = float(((1.0 - y_true) * y_pred).sum().item())
        fn = float((y_true * (1.0 - y_pred)).sum().item())
        recall = tp / max(tp + fn, 1e-8)
        precision = tp / max(tp + fp, 1e-8)
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        recalls.append(recall)
        f1s.append(f1)

        cls_scores = logits_cat[:, c]
        auc = _safe_binary_auc(cls_scores, y_true)
        ap = _safe_average_precision(cls_scores, y_true)
        if auc is not None:
            aucs.append(auc)
        if ap is not None:
            aps.append(ap)

    return {
        "top1": float(correct_top1 / n),
        "top5": float(correct_top5 / n),
        "macro_f1": _mean_or_zero(f1s),
        "balanced_acc": _mean_or_zero(recalls),
        "ovr_macro_auroc": _mean_or_zero(aucs),
        "ovr_macro_auprc": _mean_or_zero(aps),
        "n_eval": float(n),
    }


def evaluate_mvp_reg(
    model: MultiTaskModel,
    loader,
    variant_x: torch.Tensor,
    protein_x: torch.Tensor,
    gene_graph_emb: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    cosine_sum = 0.0
    abs_sum = 0.0
    sq_sum = 0.0
    n = 0
    n_elem = 0
    pred_parts: List[torch.Tensor] = []
    tgt_parts: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            variant_idx = batch["variant_idx"].to(device)
            gene_idx = batch["gene_idx"].to(device)
            target = batch["target"].to(device)

            pred = model.forward_mvp_reg(
                variant_idx,
                gene_idx,
                variant_x,
                protein_x,
                gene_graph_emb,
            )
            pred_n = F.normalize(pred, dim=-1)
            target_n = F.normalize(target, dim=-1)

            cosine = F.cosine_similarity(pred_n, target_n, dim=-1)
            diff = pred_n - target_n
            cosine_sum += float(cosine.sum().item())
            abs_sum += float(torch.abs(diff).sum().item())
            sq_sum += float((diff * diff).sum().item())
            n += pred.shape[0]
            n_elem += pred.numel()
            pred_parts.append(pred_n.reshape(-1).detach().cpu())
            tgt_parts.append(target_n.reshape(-1).detach().cpu())

    if n == 0 or n_elem == 0:
        return {
            "cosine": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "pearson": 0.0,
            "spearman": 0.0,
            "n_eval": 0.0,
        }

    pred_flat = torch.cat(pred_parts, dim=0)
    tgt_flat = torch.cat(tgt_parts, dim=0)
    pearson = _safe_pearson(pred_flat, tgt_flat)
    spearman = _safe_spearman(pred_flat, tgt_flat)

    return {
        "cosine": float(cosine_sum / n),
        "mae": float(abs_sum / n_elem),
        "rmse": float((sq_sum / n_elem) ** 0.5),
        "pearson": float(pearson if pearson is not None else 0.0),
        "spearman": float(spearman if spearman is not None else 0.0),
        "n_eval": float(n),
    }


def evaluate_func(
    model: MultiTaskModel,
    loader,
    variant_x: torch.Tensor,
    protein_x: torch.Tensor,
    gene_graph_emb: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    abs_sum = 0.0
    sq_sum = 0.0
    mask_sum = 0.0
    n = 0
    obs_pred_parts: List[torch.Tensor] = []
    obs_tgt_parts: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            variant_idx = batch["variant_idx"].to(device)
            gene_idx = batch["gene_idx"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)

            pred = model.forward_func(
                variant_idx,
                gene_idx,
                variant_x,
                protein_x,
                gene_graph_emb,
            )
            diff = pred - target
            abs_sum += float((torch.abs(diff) * mask).sum().item())
            sq_sum += float(((diff * diff) * mask).sum().item())
            mask_sum += float(mask.sum().item())
            n += pred.shape[0]
            observed = mask > 0
            if observed.any():
                obs_pred_parts.append(pred[observed].detach().cpu())
                obs_tgt_parts.append(target[observed].detach().cpu())

    if n == 0:
        return {
            "masked_mae": 0.0,
            "masked_rmse": 0.0,
            "masked_pearson": 0.0,
            "masked_spearman": 0.0,
            "n_eval": 0.0,
        }

    if obs_pred_parts:
        obs_pred = torch.cat(obs_pred_parts, dim=0)
        obs_tgt = torch.cat(obs_tgt_parts, dim=0)
        pearson = _safe_pearson(obs_pred, obs_tgt)
        spearman = _safe_spearman(obs_pred, obs_tgt)
    else:
        pearson = None
        spearman = None

    return {
        "masked_mae": float(abs_sum / max(mask_sum, 1e-8)),
        "masked_rmse": float((sq_sum / max(mask_sum, 1e-8)) ** 0.5),
        "masked_pearson": float(pearson if pearson is not None else 0.0),
        "masked_spearman": float(spearman if spearman is not None else 0.0),
        "n_eval": float(n),
    }
