from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.nn.utils import clip_grad_norm_

from eval import evaluate_domain, evaluate_func, evaluate_main, evaluate_mvp_reg
from losses import (
    domain_loss,
    func_impact_loss,
    main_multi_positive_bce_loss,
    main_multi_positive_softmax_loss,
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
    eval_graph,
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
    main_logit_scale_learnable: bool,
    main_logit_scale_min: float,
    main_logit_scale_max: float,
    domain_temperature: float,
    main_loss_type: str,
    aux_update_hgt: bool,
    aux_domain_interval: int,
    aux_mvp_interval: int,
    aux_func_interval: int,
    func_loss_type: str,
    func_smooth_l1_beta: float,
    gate_entropy_weight_start: float,
    gate_entropy_weight_end: float,
    func_column_scales: torch.Tensor,
    func_target_cols: Sequence[str],
    func_column_weights: Optional[torch.Tensor],
    device: torch.device,
    output_dir: str,
) -> Dict[str, object]:
    main_loss_type = main_loss_type.lower()
    func_loss_type = func_loss_type.lower()
    disease_ids = list(disease_ids)
    disease_id_to_col = {d: i for i, d in enumerate(disease_ids)}
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path / "best_model.pt"
    aux_domain_interval = max(int(aux_domain_interval), 1)
    aux_mvp_interval = max(int(aux_mvp_interval), 1)
    aux_func_interval = max(int(aux_func_interval), 1)

    if func_column_weights is None:
        func_column_weights = torch.ones(
            len(func_target_cols), dtype=torch.float32, device=device
        )
    else:
        func_column_weights = func_column_weights.to(device)
    func_column_scales = func_column_scales.to(device)

    best_mrr = float("-inf")
    best_state = None
    wait = 0
    history: List[Dict[str, object]] = []
    gate_temperature = 5.0

    variant_x = variant_x.to(device)
    protein_x = protein_x.to(device)
    domain_embeddings = domain_embeddings.to(device)
    graph = graph.to(device)
    eval_graph = eval_graph.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        task_iters = {
            name: iter(loader)
            for name, loader in train_loaders.items()
            if loader is not None and len(loader) > 0
        }
        if train_loaders.get("main") is not None and len(train_loaders["main"]) > 0:
            max_steps = len(train_loaders["main"])
        else:
            max_steps = max((len(loader) for loader in train_loaders.values() if loader is not None), default=0)

        epoch_loss = 0.0
        step_count = 0
        epoch_gate_entropy = 0.0
        epoch_gate_entropy_steps = 0
        progress = (epoch - 1) / max(epochs - 1, 1)
        gate_entropy_weight = (
            gate_entropy_weight_start
            + (gate_entropy_weight_end - gate_entropy_weight_start) * progress
        )

        for step_idx in range(1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)

            gene_graph_emb, trait_graph_emb = model.forward_graph(
                graph.x_dict,
                graph.edge_index_dict,
            )
            aux_gene_graph_emb = gene_graph_emb if aux_update_hgt else gene_graph_emb.detach()

            losses = {}
            gate_weight_batches: List[torch.Tensor] = []

            # Main
            if "main" in task_iters:
                try:
                    batch = next(task_iters["main"])
                except StopIteration:
                    pass
                else:
                    variant_idx = batch["variant_idx"].to(device)
                    gene_idx = batch["gene_idx"].to(device)
                    main_out = model.forward_main(
                        variant_idx,
                        gene_idx,
                        variant_x,
                        protein_x,
                        gene_graph_emb,
                        trait_graph_emb,
                        disease_ids,
                        disease_to_traits,
                        gate_temperature=gate_temperature,
                        return_gate_weights=(gate_entropy_weight > 0),
                    )
                    if gate_entropy_weight > 0:
                        z_v, z_d, gate_weights = main_out
                        gate_weight_batches.append(gate_weights)
                    else:
                        z_v, z_d = main_out
                    positives = [
                        [disease_id_to_col[d] for d in p if d in disease_id_to_col]
                        for p in batch["positive_disease_ids"]
                    ]
                    sample_weights = batch["confidence"].to(device)
                    main_logit_scale = model.get_main_logit_scale(
                        min_scale=main_logit_scale_min,
                        max_scale=main_logit_scale_max,
                    )
                    if not main_logit_scale_learnable:
                        main_logit_scale = main_logit_scale.detach()
                    if main_loss_type == "bce":
                        losses["main"] = main_multi_positive_bce_loss(
                            z_v,
                            z_d,
                            positives,
                            temperature=main_temperature,
                            logit_scale=main_logit_scale,
                            sample_weights=sample_weights,
                        )
                    else:
                        losses["main"] = main_multi_positive_softmax_loss(
                            z_v,
                            z_d,
                            positives,
                            temperature=main_temperature,
                            logit_scale=main_logit_scale,
                            sample_weights=sample_weights,
                        )

            # Domain
            if "domain" in task_iters and (step_idx % aux_domain_interval == 0):
                try:
                    batch = next(task_iters["domain"])
                except StopIteration:
                    pass
                else:
                    variant_idx = batch["variant_idx"].to(device)
                    gene_idx = batch["gene_idx"].to(device)
                    labels = batch["label"].to(device)
                    domain_out = model.forward_domain(
                        variant_idx,
                        gene_idx,
                        variant_x,
                        protein_x,
                        aux_gene_graph_emb,
                        domain_embeddings,
                        temperature=domain_temperature,
                        gate_temperature=gate_temperature,
                        return_gate_weights=(gate_entropy_weight > 0),
                    )
                    if gate_entropy_weight > 0:
                        logits, gate_weights = domain_out
                        gate_weight_batches.append(gate_weights)
                    else:
                        logits = domain_out
                    losses["domain"] = domain_loss(logits, labels)

            # MVP-REG
            if "mvp" in task_iters and (step_idx % aux_mvp_interval == 0):
                try:
                    batch = next(task_iters["mvp"])
                except StopIteration:
                    pass
                else:
                    variant_idx = batch["variant_idx"].to(device)
                    gene_idx = batch["gene_idx"].to(device)
                    target = batch["target"].to(device)
                    mvp_out = model.forward_mvp_reg(
                        variant_idx,
                        gene_idx,
                        variant_x,
                        protein_x,
                        aux_gene_graph_emb,
                        gate_temperature=gate_temperature,
                        return_gate_weights=(gate_entropy_weight > 0),
                    )
                    if gate_entropy_weight > 0:
                        pred, gate_weights = mvp_out
                        gate_weight_batches.append(gate_weights)
                    else:
                        pred = mvp_out
                    losses["mvp_reg"] = mvp_reg_loss(pred, target)

            # FuncImpact
            if "func" in task_iters and (step_idx % aux_func_interval == 0):
                try:
                    batch = next(task_iters["func"])
                except StopIteration:
                    pass
                else:
                    variant_idx = batch["variant_idx"].to(device)
                    gene_idx = batch["gene_idx"].to(device)
                    target = batch["target"].to(device)
                    mask = batch["mask"].to(device)
                    func_out = model.forward_func(
                        variant_idx,
                        gene_idx,
                        variant_x,
                        protein_x,
                        aux_gene_graph_emb,
                        gate_temperature=gate_temperature,
                        return_gate_weights=(gate_entropy_weight > 0),
                    )
                    if gate_entropy_weight > 0:
                        pred, gate_weights = func_out
                        gate_weight_batches.append(gate_weights)
                    else:
                        pred = func_out
                    losses["func"] = func_impact_loss(
                        pred,
                        target,
                        mask,
                        column_weights=func_column_weights,
                        column_scales=func_column_scales,
                        loss_type=func_loss_type,
                        smooth_l1_beta=func_smooth_l1_beta,
                    )

            if not losses:
                continue
            if gate_entropy_weight > 0 and gate_weight_batches:
                all_gate = torch.cat(gate_weight_batches, dim=0)
                gate_entropy = -(all_gate.clamp_min(1e-8) * all_gate.clamp_min(1e-8).log()).sum(dim=-1).mean()
                losses["gate_entropy_reg"] = -gate_entropy_weight * gate_entropy
                epoch_gate_entropy += float(gate_entropy.item())
                epoch_gate_entropy_steps += 1

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
            graph=eval_graph,
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
        avg_gate_entropy = epoch_gate_entropy / max(epoch_gate_entropy_steps, 1)
        main_logit_scale_val = float(
            model.get_main_logit_scale(
                min_scale=main_logit_scale_min,
                max_scale=main_logit_scale_max,
            ).detach().cpu().item()
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_epoch_loss,
                "gate_entropy": avg_gate_entropy,
                "gate_entropy_weight": gate_entropy_weight,
                "main_logit_scale": main_logit_scale_val,
                "val_metrics": val_metrics,
            }
        )

        print(
            f"epoch={epoch} train_loss={avg_epoch_loss:.4f} "
            f"logit_scale={main_logit_scale_val:.4f} "
            f"gate_entropy={avg_gate_entropy:.4f} "
            f"gate_lambda={gate_entropy_weight:.6f} "
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
