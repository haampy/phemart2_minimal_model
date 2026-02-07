from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from config import default_config, ensure_output_dir
from data import (
    apply_split,
    build_disease_to_traits_map,
    build_feature_store,
    build_global_variant_split,
    build_hetero_graph,
    build_mappings,
    compute_train_test_overlap,
    load_disease_table,
    load_domain_labels,
    load_embeddings,
    load_func_labels,
    load_main_labels,
    load_mvp_reg_targets,
    make_dataloader_for_task,
    make_domain_records,
    make_func_records,
    make_main_records,
    make_mvp_records,
    normalize_id,
    summarize_split,
)
from model import MultiTaskModel
from train import evaluate_all_tasks, train_multitask


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = out.index.map(normalize_id)
    out = out[~out.index.duplicated(keep="first")]
    return out


def load_domain_embedding_tensor(path: str, num_domains: int, embedding_dim: int) -> torch.Tensor:
    df = pd.read_csv(path, index_col=0)
    if df.shape[1] != embedding_dim:
        # Fallback when file has no index column.
        df2 = pd.read_csv(path)
        numeric_cols = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]
        if len(numeric_cols) < embedding_dim:
            raise ValueError(
                f"Domain embedding width mismatch: got {df.shape[1]}, expected {embedding_dim}"
            )
        df = df2[numeric_cols[:embedding_dim]]

    arr = df.to_numpy(dtype=np.float32)
    if arr.shape[0] < num_domains:
        raise ValueError(f"Domain embedding rows ({arr.shape[0]}) < num_domains ({num_domains})")
    arr = arr[:num_domains, :embedding_dim]
    return torch.tensor(arr, dtype=torch.float32)


def fill_missing_gene_ids(target_df: pd.DataFrame, refs: List[pd.DataFrame]) -> pd.DataFrame:
    out = target_df.copy()
    lookup: Dict[str, str] = {}
    for ref in refs:
        if "variant_id" not in ref.columns or "gene_id" not in ref.columns:
            continue
        sub = ref[["variant_id", "gene_id"]].dropna()
        for row in sub.itertuples(index=False):
            variant_id, gene_id = row
            if variant_id and gene_id and variant_id not in lookup:
                lookup[variant_id] = gene_id

    missing_before = int((out["gene_id"] == "").sum())
    if missing_before == 0:
        return out

    out.loc[out["gene_id"] == "", "gene_id"] = (
        out.loc[out["gene_id"] == "", "variant_id"].map(lookup).fillna("")
    )
    missing_after = int((out["gene_id"] == "").sum())
    print(f"mvp_missing_gene_before={missing_before} after_fill={missing_after}")
    return out


def apply_mvp_gene_map(target_df: pd.DataFrame, gene_map_csv: str) -> pd.DataFrame:
    out = target_df.copy()
    map_path = Path(gene_map_csv)
    if not map_path.exists():
        print(f"mvp_gene_map_not_found={map_path}")
        return out

    map_df = pd.read_csv(map_path)
    if "variant_id" not in map_df.columns or "gene_id" not in map_df.columns:
        raise ValueError(f"MVP gene map must contain variant_id and gene_id: {map_path}")

    map_df["variant_id"] = map_df["variant_id"].map(normalize_id)
    map_df["gene_id"] = map_df["gene_id"].map(normalize_id)
    map_df = map_df[(map_df["variant_id"] != "") & (map_df["gene_id"] != "")]
    map_dict = dict(zip(map_df["variant_id"], map_df["gene_id"]))

    missing_before = int((out["gene_id"] == "").sum())
    out.loc[out["gene_id"] == "", "gene_id"] = (
        out.loc[out["gene_id"] == "", "variant_id"].map(map_dict).fillna("")
    )
    missing_after = int((out["gene_id"] == "").sum())
    print(f"mvp_gene_map_fill_before={missing_before} after={missing_after}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal PheMART2 multi-task runner")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config()

    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.device is not None:
        cfg.runtime.device = args.device
    if args.seed is not None:
        cfg.split.seed = args.seed
    if args.output_dir is not None:
        cfg.paths.output_dir = args.output_dir

    set_seed(cfg.split.seed)
    out_dir = ensure_output_dir(cfg)

    device = torch.device(cfg.runtime.device)
    print(f"device={device}")

    print("[1/7] loading labels")
    main_df = load_main_labels(cfg.paths.main_labels)
    disease_df = load_disease_table(cfg.paths.disease_table)
    domain_df = load_domain_labels(cfg.paths.domain_labels)
    func_df = load_func_labels(cfg.paths.func_labels)
    mvp_df = load_mvp_reg_targets(cfg.paths.mvp_variant_map, cfg.paths.mvp_targets)
    mvp_df = apply_mvp_gene_map(mvp_df, cfg.paths.mvp_variant_gene_map)
    mvp_df = fill_missing_gene_ids(mvp_df, refs=[domain_df, func_df, main_df])
    unresolved_before_drop = int((mvp_df["gene_id"] == "").sum())
    if unresolved_before_drop > 0:
        mvp_df = mvp_df[mvp_df["gene_id"] != ""].copy()
        print(f"mvp_drop_unresolved_gene_rows={unresolved_before_drop}")

    print("[2/7] building global split")
    split_map = build_global_variant_split(
        main_df=main_df,
        domain_df=domain_df,
        mvp_df=mvp_df[["variant_id", "gene_id"]],
        func_df=func_df,
        seed=cfg.split.seed,
        ratios=(cfg.split.train_ratio, cfg.split.val_ratio, cfg.split.test_ratio),
    )

    main_split = apply_split(main_df, "variant_id", split_map)
    domain_split = apply_split(domain_df, "variant_id", split_map)
    mvp_split = apply_split(mvp_df, "variant_id", split_map)
    func_split = apply_split(func_df, "variant_id", split_map)

    split_summary = summarize_split(
        split_map,
        {
            "main": main_df,
            "domain": domain_df,
            "mvp": mvp_df,
            "func": func_df,
        },
    )
    overlap = compute_train_test_overlap(
        {
            "main": main_split,
            "domain": domain_split,
            "mvp": mvp_split,
            "func": func_split,
        }
    )

    print("split_summary=" + json.dumps(split_summary, ensure_ascii=False))
    print("cross_task_overlap=" + json.dumps(overlap, ensure_ascii=False))

    print("[3/7] loading graph data")
    gene_x_df = _normalize_index(load_embeddings(cfg.paths.gene_x))
    trait_x_df = _normalize_index(load_embeddings(cfg.paths.trait_x))
    mappings = build_mappings(gene_x_df, trait_x_df, disease_df)

    graph = build_hetero_graph(
        gene_x_df=gene_x_df,
        trait_x_df=trait_x_df,
        edge_files={
            "gene_to_gene": cfg.paths.gene_to_gene,
            "gene_to_trait": cfg.paths.gene_to_trait,
            "trait_to_trait": cfg.paths.trait_to_trait,
        },
        gene_mapping=mappings["gene_to_idx"],
        trait_mapping=mappings["trait_to_idx"],
    )

    disease_to_traits = build_disease_to_traits_map(disease_df, mappings["trait_to_idx"])
    all_disease_ids: List[int] = sorted(set(disease_df["disease_index"].tolist()) & set(disease_to_traits.keys()))

    print("[4/7] loading variant/protein embeddings")
    required_variants = set()
    for split in [main_split, domain_split, mvp_split, func_split]:
        for part in split:
            required_variants.update(part["variant_id"].tolist())

    variant_x_df = _normalize_index(load_embeddings(cfg.paths.variant_x, required_ids=required_variants))
    protein_x_df = _normalize_index(load_embeddings(cfg.paths.protein_x, required_ids=required_variants))
    feature_store = build_feature_store(variant_x_df, protein_x_df)

    print("[5/7] preparing dataloaders")
    main_train, main_val, main_test = main_split
    domain_train, domain_val, domain_test = domain_split
    mvp_train, mvp_val, mvp_test = mvp_split
    func_train, func_val, func_test = func_split

    mvp_target_cols = sorted([c for c in mvp_df.columns if c.startswith("mvp_")], key=lambda x: int(x.split("_")[1]))

    records = {
        "train": {
            "main": make_main_records(main_train, feature_store.variant_to_idx, mappings["gene_to_idx"]),
            "domain": make_domain_records(domain_train, feature_store.variant_to_idx, mappings["gene_to_idx"]),
            "mvp": make_mvp_records(
                mvp_train,
                feature_store.variant_to_idx,
                mappings["gene_to_idx"],
                mvp_target_cols,
            ),
            "func": make_func_records(func_train, feature_store.variant_to_idx, mappings["gene_to_idx"]),
        },
        "val": {
            "main": make_main_records(main_val, feature_store.variant_to_idx, mappings["gene_to_idx"]),
            "domain": make_domain_records(domain_val, feature_store.variant_to_idx, mappings["gene_to_idx"]),
            "mvp": make_mvp_records(
                mvp_val,
                feature_store.variant_to_idx,
                mappings["gene_to_idx"],
                mvp_target_cols,
            ),
            "func": make_func_records(func_val, feature_store.variant_to_idx, mappings["gene_to_idx"]),
        },
        "test": {
            "main": make_main_records(main_test, feature_store.variant_to_idx, mappings["gene_to_idx"]),
            "domain": make_domain_records(domain_test, feature_store.variant_to_idx, mappings["gene_to_idx"]),
            "mvp": make_mvp_records(
                mvp_test,
                feature_store.variant_to_idx,
                mappings["gene_to_idx"],
                mvp_target_cols,
            ),
            "func": make_func_records(func_test, feature_store.variant_to_idx, mappings["gene_to_idx"]),
        },
    }

    for split_name, task_records in records.items():
        counts = {k: len(v) for k, v in task_records.items()}
        print(f"records_{split_name}=" + json.dumps(counts, ensure_ascii=False))

    train_loaders = {
        "main": make_dataloader_for_task(
            "main",
            records["train"]["main"],
            batch_size=cfg.train.batch_size_main,
            shuffle=True,
            num_workers=cfg.runtime.num_workers,
        ),
        "domain": make_dataloader_for_task(
            "domain",
            records["train"]["domain"],
            batch_size=cfg.train.batch_size_domain,
            shuffle=True,
            num_workers=cfg.runtime.num_workers,
        ),
        "mvp": make_dataloader_for_task(
            "mvp",
            records["train"]["mvp"],
            batch_size=cfg.train.batch_size_mvp,
            shuffle=True,
            num_workers=cfg.runtime.num_workers,
        ),
        "func": make_dataloader_for_task(
            "func",
            records["train"]["func"],
            batch_size=cfg.train.batch_size_func,
            shuffle=True,
            num_workers=cfg.runtime.num_workers,
        ),
    }
    val_loaders = {
        "main": make_dataloader_for_task("main", records["val"]["main"], cfg.train.batch_size_main, False),
        "domain": make_dataloader_for_task("domain", records["val"]["domain"], cfg.train.batch_size_domain, False),
        "mvp": make_dataloader_for_task("mvp", records["val"]["mvp"], cfg.train.batch_size_mvp, False),
        "func": make_dataloader_for_task("func", records["val"]["func"], cfg.train.batch_size_func, False),
    }
    test_loaders = {
        "main": make_dataloader_for_task("main", records["test"]["main"], cfg.train.batch_size_main, False),
        "domain": make_dataloader_for_task("domain", records["test"]["domain"], cfg.train.batch_size_domain, False),
        "mvp": make_dataloader_for_task("mvp", records["test"]["mvp"], cfg.train.batch_size_mvp, False),
        "func": make_dataloader_for_task("func", records["test"]["func"], cfg.train.batch_size_func, False),
    }

    domain_embeddings = load_domain_embedding_tensor(
        cfg.paths.domain_embeddings,
        num_domains=cfg.model.num_domains,
        embedding_dim=cfg.model.domain_embedding_dim,
    )

    print("[6/7] init model + train")
    model = MultiTaskModel(
        metadata=graph.metadata(),
        gene_in_dim=cfg.model.graph_gene_in_dim,
        trait_in_dim=cfg.model.graph_trait_in_dim,
        variant_in_dim=cfg.model.variant_in_dim,
        protein_in_dim=cfg.model.protein_in_dim,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.out_dim,
        num_heads=cfg.model.num_heads,
        num_graph_layers=cfg.model.num_graph_layers,
        dropout=cfg.model.dropout,
        num_domains=cfg.model.num_domains,
        domain_embedding_dim=cfg.model.domain_embedding_dim,
        mvp_out_dim=cfg.model.mvp_out_dim,
        func_out_dim=cfg.model.func_out_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-6,
    )

    train_result = train_multitask(
        model=model,
        graph=graph,
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        variant_x=feature_store.variant_x,
        protein_x=feature_store.protein_x,
        domain_embeddings=domain_embeddings,
        disease_ids=all_disease_ids,
        disease_to_traits=disease_to_traits,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_weights={
            "main": cfg.loss_weights.main,
            "domain": cfg.loss_weights.domain,
            "mvp_reg": cfg.loss_weights.mvp_reg,
            "func": cfg.loss_weights.func,
        },
        epochs=cfg.train.epochs,
        grad_clip_norm=cfg.train.grad_clip_norm,
        early_stopping_patience=cfg.train.early_stopping_patience,
        main_temperature=cfg.train.main_temperature,
        domain_temperature=cfg.train.domain_temperature,
        device=device,
        output_dir=str(out_dir),
    )

    print("train_result=" + json.dumps({"best_val_mrr": train_result["best_val_mrr"]}, ensure_ascii=False))

    print("[7/7] final test eval")
    test_metrics = evaluate_all_tasks(
        model=model,
        graph=graph.to(device),
        loaders=test_loaders,
        variant_x=feature_store.variant_x.to(device),
        protein_x=feature_store.protein_x.to(device),
        domain_embeddings=domain_embeddings.to(device),
        disease_ids=all_disease_ids,
        disease_to_traits=disease_to_traits,
        device=device,
        domain_temperature=cfg.train.domain_temperature,
    )

    print("test_main=" + json.dumps(test_metrics.get("main", {}), ensure_ascii=False))
    print("test_domain=" + json.dumps(test_metrics.get("domain", {}), ensure_ascii=False))
    print("test_mvp=" + json.dumps(test_metrics.get("mvp", {}), ensure_ascii=False))
    print("test_func=" + json.dumps(test_metrics.get("func", {}), ensure_ascii=False))

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
