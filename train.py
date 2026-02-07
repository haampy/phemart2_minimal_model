from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.nn.utils import clip_grad_norm_

from eval import evaluate_domain, evaluate_func, evaluate_main, evaluate_mvp_reg
from losses import (
    domain_loss,
    func_impact_loss,
    main_multi_positive_bce_loss,
    mvp_reg_loss,
    total_loss,
)
from model import MultiTaskModel


def evaluate_all_tasks(
    model: MultiTaskModel,
    graph,
    loaders: Dict[str, object],
    variant_x: torch.Tensor,
    protein_x: torch.Tensor,
    domain_embeddings: torch.Tensor,
    disease_ids: Sequence[int],
    disease_to_traits: Dict[int, List[int]],
    device: torch.device,
    domain_temperature: float,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    with torch.no_grad():
        gene_graph_emb, trait_graph_emb = model.forward_graph(
            graph.x_dict,
            graph.edge_index_dict,
        )

    out: Dict[str, Dict[str, float]] = {}
    if loaders.get("main") is not None and len(loaders["main"]) > 0:
        out["main"] = evaluate_main(
            model,
            loaders["main"],
            variant_x,
            protein_x,
            gene_graph_emb,
            trait_graph_emb,
            disease_ids,
            disease_to_traits,
            device,
        )
    if loaders.get("domain") is not None and len(loaders["domain"]) > 0:
        out["domain"] = evaluate_domain(
            model,
            loaders["domain"],
            variant_x,
            protein_x,
            gene_graph_emb,
            domain_embeddings,
            device,
            temperature=domain_temperature,
        )
    if loaders.get("mvp") is not None and len(loaders["mvp"]) > 0:
        out["mvp"] = evaluate_mvp_reg(
            model,
            loaders["mvp"],
            variant_x,
            protein_x,
            gene_graph_emb,
            device,
        )
    if loaders.get("func") is not None and len(loaders["func"]) > 0:
        out["func"] = evaluate_func(
            model,
            loaders["func"],
            variant_x,
            protein_x,
            gene_graph_emb,
            device,
        )
    return out


def train_multitask(
    model: MultiTaskModel,
    graph,
    train_loaders: Dict[str, object],
    val_loaders: Dict[str, object],
    variant_x: torch.Tensor,
    protein_x: torch.Tensor,
    domain_embeddings: torch.Tensor,
    disease_ids: Sequence[int],
    disease_to_traits: Dict[int, List[int]],
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_weights: Dict[str, float],
    epochs: int,
    grad_clip_norm: float,
    early_stopping_patience: int,
    main_temperature: float,
    domain_temperature: float,
    device: torch.device,
    output_dir: str,
) -> Dict[str, object]:
    disease_ids = list(disease_ids)
    disease_id_to_col = {d: i for i, d in enumerate(disease_ids)}
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path / "best_model.pt"

    func_column_weights = torch.tensor(
        [1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5],
        dtype=torch.float32,
        device=device,
    )

    best_mrr = float("-inf")
    best_state = None
    wait = 0
    history: List[Dict[str, object]] = []
    gate_temperature = 5.0

    variant_x = variant_x.to(device)
    protein_x = protein_x.to(device)
    domain_embeddings = domain_embeddings.to(device)
    graph = graph.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        task_iters = {
            name: iter(loader)
            for name, loader in train_loaders.items()
            if loader is not None and len(loader) > 0
        }
        max_steps = max((len(loader) for loader in train_loaders.values() if loader is not None), default=0)

        epoch_loss = 0.0
        step_count = 0

        for _ in range(max_steps):
            optimizer.zero_grad(set_to_none=True)

            gene_graph_emb, trait_graph_emb = model.forward_graph(
                graph.x_dict,
                graph.edge_index_dict,
            )

            losses = {}

            # Main
            if "main" in task_iters:
                try:
                    batch = next(task_iters["main"])
                except StopIteration:
                    pass
                else:
                    variant_idx = batch["variant_idx"].to(device)
                    gene_idx = batch["gene_idx"].to(device)
                    z_v, z_d = model.forward_main(
                        variant_idx,
                        gene_idx,
                        variant_x,
                        protein_x,
                        gene_graph_emb,
                        trait_graph_emb,
                        disease_ids,
                        disease_to_traits,
                        gate_temperature=gate_temperature,
                    )
                    positives = [
                        [disease_id_to_col[d] for d in p if d in disease_id_to_col]
                        for p in batch["positive_disease_ids"]
                    ]
                    losses["main"] = main_multi_positive_bce_loss(
                        z_v,
                        z_d,
                        positives,
                        temperature=main_temperature,
                    )

            # Domain
            if "domain" in task_iters:
                try:
                    batch = next(task_iters["domain"])
                except StopIteration:
                    pass
                else:
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
                        temperature=domain_temperature,
                        gate_temperature=gate_temperature,
                    )
                    losses["domain"] = domain_loss(logits, labels)

            # MVP-REG
            if "mvp" in task_iters:
                try:
                    batch = next(task_iters["mvp"])
                except StopIteration:
                    pass
                else:
                    variant_idx = batch["variant_idx"].to(device)
                    gene_idx = batch["gene_idx"].to(device)
                    target = batch["target"].to(device)
                    pred = model.forward_mvp_reg(
                        variant_idx,
                        gene_idx,
                        variant_x,
                        protein_x,
                        gene_graph_emb,
                        gate_temperature=gate_temperature,
                    )
                    losses["mvp_reg"] = mvp_reg_loss(pred, target)

            # FuncImpact
            if "func" in task_iters:
                try:
                    batch = next(task_iters["func"])
                except StopIteration:
                    pass
                else:
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
                        gate_temperature=gate_temperature,
                    )
                    losses["func"] = func_impact_loss(
                        pred,
                        target,
                        mask,
                        column_weights=func_column_weights,
                    )

            if not losses:
                continue

            loss = total_loss(losses, loss_weights)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            epoch_loss += float(loss.item())
            step_count += 1

        if scheduler is not None:
            scheduler.step()

        val_metrics = evaluate_all_tasks(
            model=model,
            graph=graph,
            loaders=val_loaders,
            variant_x=variant_x,
            protein_x=protein_x,
            domain_embeddings=domain_embeddings,
            disease_ids=disease_ids,
            disease_to_traits=disease_to_traits,
            device=device,
            domain_temperature=domain_temperature,
        )

        val_mrr = val_metrics.get("main", {}).get("mrr", 0.0)
        avg_epoch_loss = epoch_loss / max(step_count, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_epoch_loss,
                "val_metrics": val_metrics,
            }
        )

        print(
            f"epoch={epoch} train_loss={avg_epoch_loss:.4f} "
            f"val_mrr={val_mrr:.4f} "
            f"val_r1={val_metrics.get('main', {}).get('recall@1', 0.0):.4f} "
            f"val_r5={val_metrics.get('main', {}).get('recall@5', 0.0):.4f} "
            f"val_r10={val_metrics.get('main', {}).get('recall@10', 0.0):.4f}"
        )

        if val_mrr > best_mrr:
            best_mrr = val_mrr
            wait = 0
            best_state = deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "epoch": epoch,
                    "val_mrr": best_mrr,
                },
                ckpt_path,
            )
        else:
            wait += 1

        gate_temperature = max(gate_temperature * 0.9, 0.1)

        if wait >= early_stopping_patience:
            print(f"early_stop_at_epoch={epoch} best_val_mrr={best_mrr:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_mrr": best_mrr,
        "history": history,
        "checkpoint_path": str(ckpt_path),
    }
