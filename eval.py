from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from model import MultiTaskModel


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
                n += 1

    if n == 0:
        return {
            "mrr": 0.0,
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "n_eval": 0.0,
        }

    return {
        "mrr": float(sum(reciprocal_ranks) / n),
        "recall@1": float(recall_counts[1] / n),
        "recall@5": float(recall_counts[5] / n),
        "recall@10": float(recall_counts[10] / n),
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

    if n == 0:
        return {"top1": 0.0, "top5": 0.0, "n_eval": 0.0}

    return {
        "top1": float(correct_top1 / n),
        "top5": float(correct_top5 / n),
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
    n = 0
    n_elem = 0

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
            cosine_sum += float(cosine.sum().item())
            abs_sum += float(torch.abs(pred_n - target_n).sum().item())
            n += pred.shape[0]
            n_elem += pred.numel()

    if n == 0 or n_elem == 0:
        return {"cosine": 0.0, "mae": 0.0, "n_eval": 0.0}

    return {
        "cosine": float(cosine_sum / n),
        "mae": float(abs_sum / n_elem),
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
    mask_sum = 0.0
    n = 0

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
            abs_sum += float((torch.abs(pred - target) * mask).sum().item())
            mask_sum += float(mask.sum().item())
            n += pred.shape[0]

    if n == 0:
        return {"masked_mae": 0.0, "n_eval": 0.0}

    return {
        "masked_mae": float(abs_sum / max(mask_sum, 1e-8)),
        "n_eval": float(n),
    }
