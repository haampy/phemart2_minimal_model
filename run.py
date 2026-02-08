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
    FUNC_TARGET_COLS,
    apply_split,
    build_rsid_to_hgvs_map,
    build_disease_to_traits_map,
    build_feature_store,
    build_global_variant_split,
    build_hetero_graph,
    build_inductive_train_graph,
    build_mappings,
    compute_func_target_scales,
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
    get_func_mask_cols,
    make_main_records,
    make_mvp_records,
    normalize_id,
    remap_variant_ids_to_hgvs,
    select_func_train_subset,
    summarize_split,
)
from model import MultiTaskModel
from train import evaluate_all_tasks, train_multitask


FUNC_DEFAULT_WEIGHT_MAP = {
    "CADD_phred": 1.0,
    "phyloP": 1.0,
    "GERP++": 1.0,
    "SIFT": 0.5,
    "Polyphen2_HDIV": 0.5,
    "MetaSVM": 1.0,
    "REVEL": 0.5,
    "AlphaMissense": 0.5,
}


TASK_MODE_TO_ENABLED = {
    "main_only": {"main"},
    "main_domain": {"main", "domain"},
    "main_domain_mvp": {"main", "domain", "mvp"},
    "main_domain_func": {"main", "domain", "func"},
    "full": {"main", "domain", "mvp", "func"},
}


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


def remap_domain_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, List[int]]:
    out = df.copy()
    raw_labels = sorted(set(out["domain_map"].astype(int).tolist()))
    if not raw_labels:
        raise ValueError("Domain label table is empty")
    label_to_idx = {raw: idx for idx, raw in enumerate(raw_labels)}
    out["domain_map"] = out["domain_map"].map(label_to_idx).astype(int)
    return out, raw_labels


def load_domain_embedding_tensor(
    path: str,
    raw_label_ids: List[int],
    embedding_dim: int,
) -> torch.Tensor:
    num_domains = len(raw_label_ids)
    df = pd.read_csv(path, index_col=0)
    try:
        df.index = pd.to_numeric(df.index, errors="coerce")
    except Exception:
        pass

    if df.shape[1] != embedding_dim:
        # Fallback when file has no index column.
        df2 = pd.read_csv(path)
        numeric_cols = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]
        if len(numeric_cols) < embedding_dim:
            raise ValueError(
                f"Domain embedding width mismatch: got {df.shape[1]}, expected {embedding_dim}"
            )
        df = df2[numeric_cols[:embedding_dim]]

    if pd.api.types.is_numeric_dtype(df.index):
        missing = [lab for lab in raw_label_ids if lab not in set(df.index.astype(int).tolist())]
        if missing:
            raise ValueError(
                f"Domain embedding file missing raw labels: {missing[:10]}"
            )
        arr = df.loc[raw_label_ids, df.columns[:embedding_dim]].to_numpy(dtype=np.float32)
    else:
        arr = df.to_numpy(dtype=np.float32)
        if arr.shape[0] < num_domains:
            raise ValueError(
                f"Domain embedding rows ({arr.shape[0]}) < num_domains ({num_domains})"
            )
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
    parser.add_argument(
        "--task-mode",
        type=str,
        choices=sorted(TASK_MODE_TO_ENABLED.keys()),
        default=None,
        help="Basic task combinations: main_only/main_domain/main_domain_mvp/main_domain_func/full",
    )
    parser.add_argument("--main-loss-type", type=str, choices=["softmax", "bce"], default=None)
    parser.add_argument("--main-logit-scale-learnable", type=int, choices=[0, 1], default=None)
    parser.add_argument("--main-logit-scale-min", type=float, default=None)
    parser.add_argument("--main-logit-scale-max", type=float, default=None)
    parser.add_argument("--aux-update-hgt", type=int, choices=[0, 1], default=None)
    parser.add_argument("--aux-domain-interval", type=int, default=None)
    parser.add_argument("--aux-mvp-interval", type=int, default=None)
    parser.add_argument("--aux-func-interval", type=int, default=None)
    parser.add_argument("--gate-entropy-weight-start", type=float, default=None)
    parser.add_argument("--gate-entropy-weight-end", type=float, default=None)
    parser.add_argument("--use-inductive-graph-train", type=int, choices=[0, 1], default=None)
    parser.add_argument(
        "--func-active-scores",
        type=str,
        default=None,
        help="Comma-separated FUNC score columns, e.g. CADD_phred,phyloP,GERP++,MetaSVM",
    )
    parser.add_argument("--func-min-valid-scores", type=int, default=None)
    parser.add_argument("--func-train-per-gene-cap", type=int, default=None)
    parser.add_argument("--min-train-records-mvp", type=int, default=None)
    parser.add_argument("--min-train-records-func", type=int, default=None)
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
    task_mode = args.task_mode or "full"
    enabled_tasks = TASK_MODE_TO_ENABLED[task_mode]
    if args.main_loss_type is not None:
        cfg.train.main_loss_type = args.main_loss_type
    if args.main_logit_scale_learnable is not None:
        cfg.train.main_logit_scale_learnable = bool(args.main_logit_scale_learnable)
    if args.main_logit_scale_min is not None:
        cfg.train.main_logit_scale_min = args.main_logit_scale_min
    if args.main_logit_scale_max is not None:
        cfg.train.main_logit_scale_max = args.main_logit_scale_max
    if args.aux_update_hgt is not None:
        cfg.train.aux_update_hgt = bool(args.aux_update_hgt)
    if args.aux_domain_interval is not None:
        cfg.train.aux_domain_interval = args.aux_domain_interval
    if args.aux_mvp_interval is not None:
        cfg.train.aux_mvp_interval = args.aux_mvp_interval
    if args.aux_func_interval is not None:
        cfg.train.aux_func_interval = args.aux_func_interval
    if args.gate_entropy_weight_start is not None:
        cfg.train.gate_entropy_weight_start = args.gate_entropy_weight_start
    if args.gate_entropy_weight_end is not None:
        cfg.train.gate_entropy_weight_end = args.gate_entropy_weight_end
    if args.use_inductive_graph_train is not None:
        cfg.train.use_inductive_graph_train = bool(args.use_inductive_graph_train)
    if args.func_active_scores is not None:
        scores = [s.strip() for s in args.func_active_scores.split(",") if s.strip()]
        cfg.train.func_active_scores = tuple(scores)
    if args.func_min_valid_scores is not None:
        cfg.train.func_min_valid_scores = args.func_min_valid_scores
    if args.func_train_per_gene_cap is not None:
        cfg.train.func_train_per_gene_cap = args.func_train_per_gene_cap
    if args.min_train_records_mvp is not None:
        cfg.train.min_train_records_mvp = args.min_train_records_mvp
    if args.min_train_records_func is not None:
        cfg.train.min_train_records_func = args.min_train_records_func
    if cfg.train.main_logit_scale_min <= 0:
        raise ValueError("main_logit_scale_min must be > 0")
    if cfg.train.main_logit_scale_max < cfg.train.main_logit_scale_min:
        raise ValueError("main_logit_scale_max must be >= main_logit_scale_min")
    for interval_name, interval_value in [
        ("aux_domain_interval", cfg.train.aux_domain_interval),
        ("aux_mvp_interval", cfg.train.aux_mvp_interval),
        ("aux_func_interval", cfg.train.aux_func_interval),
    ]:
        if interval_value < 1:
            raise ValueError(f"{interval_name} must be >= 1")
    print(f"task_mode={task_mode} enabled_tasks={sorted(enabled_tasks)}")

    set_seed(cfg.split.seed)
    out_dir = ensure_output_dir(cfg)

    device = torch.device(cfg.runtime.device)
    print(f"device={device}")

    print("[1/7] loading labels")
    main_df = load_main_labels(cfg.paths.main_labels)
    disease_df = load_disease_table(cfg.paths.disease_table)
    domain_df = load_domain_labels(cfg.paths.domain_labels)
    func_target_cols = list(cfg.train.func_active_scores)
    unknown_func_targets = [c for c in func_target_cols if c not in FUNC_TARGET_COLS]
    if unknown_func_targets:
        raise ValueError(f"Unknown FUNC target columns: {unknown_func_targets}")
    if len(func_target_cols) == 0:
        raise ValueError("func_active_scores must not be empty")
    if cfg.train.func_min_valid_scores > len(func_target_cols):
        raise ValueError(
            f"func_min_valid_scores ({cfg.train.func_min_valid_scores}) exceeds "
            f"active FUNC targets ({len(func_target_cols)})"
        )
    func_mask_cols = get_func_mask_cols(func_target_cols)
    print("func_active_scores=" + json.dumps(func_target_cols, ensure_ascii=False))

    domain_df, raw_domain_labels = remap_domain_labels(domain_df)
    num_domains = len(raw_domain_labels)
    print(
        f"domain_label_remap=num_domains={num_domains} "
        f"raw_min={raw_domain_labels[0]} raw_max={raw_domain_labels[-1]}"
    )
    func_df = load_func_labels(cfg.paths.func_labels, target_cols=func_target_cols)
    if "mvp" in enabled_tasks:
        mvp_df = load_mvp_reg_targets(cfg.paths.mvp_variant_map, cfg.paths.mvp_targets)
        mvp_df = apply_mvp_gene_map(mvp_df, cfg.paths.mvp_variant_gene_map)
        mvp_df = fill_missing_gene_ids(mvp_df, refs=[domain_df, func_df, main_df])
        unresolved_before_drop = int((mvp_df["gene_id"] == "").sum())
        if unresolved_before_drop > 0:
            mvp_df = mvp_df[mvp_df["gene_id"] != ""].copy()
            print(f"mvp_drop_unresolved_gene_rows={unresolved_before_drop}")
    else:
        mvp_df = pd.DataFrame(columns=["variant_id", "gene_id"])
        print("mvp_task_disabled=1")

    if "mvp" in enabled_tasks or "func" in enabled_tasks:
        rsid_to_hgvs = build_rsid_to_hgvs_map(
            hgvs_embed_csv=cfg.paths.mvp_hgvs_embeddings,
            rsid_embed_csv=cfg.paths.mvp_rsid_embeddings,
        )
        print(f"rsid_to_hgvs_size={len(rsid_to_hgvs)}")
        if "mvp" in enabled_tasks:
            mvp_df = remap_variant_ids_to_hgvs(mvp_df, rsid_to_hgvs, task_name="mvp")
        if "func" in enabled_tasks:
            func_df = remap_variant_ids_to_hgvs(func_df, rsid_to_hgvs, task_name="func")
    else:
        print("rsid_to_hgvs_skip=1")

    if "mvp" in enabled_tasks:
        mvp_before_dedup = len(mvp_df)
        mvp_df = mvp_df.drop_duplicates(subset=["variant_id"], keep="first").copy()
        print(f"mvp_dedup_drop={mvp_before_dedup - len(mvp_df)}")
    func_before_dedup = len(func_df)
    func_df = func_df.drop_duplicates(subset=["variant_id", "gene_id"], keep="first").copy()
    print(f"func_dedup_drop={func_before_dedup - len(func_df)}")

    print("[2/7] building global split")
    split_map, gene_split_map = build_global_variant_split(
        main_df=main_df,
        domain_df=domain_df,
        mvp_df=mvp_df[["variant_id", "gene_id"]],
        func_df=func_df,
        seed=cfg.split.seed,
        ratios=(cfg.split.train_ratio, cfg.split.val_ratio, cfg.split.test_ratio),
        return_gene_split=True,
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
    if cfg.train.use_inductive_graph_train:
        train_gene_indices = {
            mappings["gene_to_idx"][g]
            for g, split_name in gene_split_map.items()
            if split_name == "train" and g in mappings["gene_to_idx"]
        }
        train_graph = build_inductive_train_graph(graph, train_gene_indices)
        print(
            "graph_mode=inductive "
            f"train_genes={len(train_gene_indices)} "
            f"full_gene_edges={graph[('gene', 'to', 'gene')].edge_index.shape[1]} "
            f"train_gene_edges={train_graph[('gene', 'to', 'gene')].edge_index.shape[1]}"
        )
    else:
        train_graph = graph
        print("graph_mode=transductive")

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
    if "func" in enabled_tasks:
        func_train = select_func_train_subset(
            func_train,
            min_valid_scores=cfg.train.func_min_valid_scores,
            per_gene_cap=cfg.train.func_train_per_gene_cap,
            seed=cfg.split.seed,
        )
    else:
        func_train = func_train.iloc[0:0].copy()
        func_val = func_val.iloc[0:0].copy()
        func_test = func_test.iloc[0:0].copy()

    mvp_target_cols = sorted([c for c in mvp_df.columns if c.startswith("mvp_")], key=lambda x: int(x.split("_")[1]))
    if "func" in enabled_tasks:
        func_target_scales = compute_func_target_scales(func_train, target_cols=func_target_cols)
    else:
        func_target_scales = np.ones(len(func_target_cols), dtype=np.float32)
    print("func_target_scales=" + json.dumps(func_target_scales.tolist(), ensure_ascii=False))

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
            "func": make_func_records(
                func_train,
                feature_store.variant_to_idx,
                mappings["gene_to_idx"],
                target_cols=func_target_cols,
                mask_cols=func_mask_cols,
            ),
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
            "func": make_func_records(
                func_val,
                feature_store.variant_to_idx,
                mappings["gene_to_idx"],
                target_cols=func_target_cols,
                mask_cols=func_mask_cols,
            ),
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
            "func": make_func_records(
                func_test,
                feature_store.variant_to_idx,
                mappings["gene_to_idx"],
                target_cols=func_target_cols,
                mask_cols=func_mask_cols,
            ),
        },
    }

    for split_name in ["train", "val", "test"]:
        for task_name in ["domain", "mvp", "func"]:
            if task_name not in enabled_tasks:
                records[split_name][task_name] = []

    for split_name, task_records in records.items():
        counts = {k: len(v) for k, v in task_records.items()}
        print(f"records_{split_name}=" + json.dumps(counts, ensure_ascii=False))
        if task_records["domain"]:
            labels = [int(r["label"]) for r in task_records["domain"]]
            bad = [y for y in labels if y < 0 or y >= num_domains]
            if bad:
                raise ValueError(
                    f"Domain labels out of range in {split_name}: "
                    f"min={min(labels)} max={max(labels)} num_domains={num_domains}"
                )

    if "mvp" in enabled_tasks:
        min_mvp = cfg.train.min_train_records_mvp
        train_mvp_n = len(records["train"]["mvp"])
        if train_mvp_n < min_mvp:
            raise ValueError(
                f"Insufficient MVP train records: mvp={train_mvp_n} (min={min_mvp})"
            )
    if "func" in enabled_tasks:
        min_func = cfg.train.min_train_records_func
        train_func_n = len(records["train"]["func"])
        if train_func_n < min_func:
            raise ValueError(
                f"Insufficient FUNC train records: func={train_func_n} (min={min_func})"
            )

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
        raw_label_ids=raw_domain_labels,
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
        num_domains=num_domains,
        domain_embedding_dim=cfg.model.domain_embedding_dim,
        mvp_out_dim=cfg.model.mvp_out_dim,
        func_out_dim=len(func_target_cols),
        modality_drop_variant=cfg.model.modality_drop_variant,
        modality_drop_protein=cfg.model.modality_drop_protein,
        modality_drop_gene=cfg.model.modality_drop_gene,
        main_temperature=cfg.train.main_temperature,
        main_logit_scale_learnable=cfg.train.main_logit_scale_learnable,
    ).to(device)

    graph_params = list(model.graph_encoder.parameters()) + list(model.disease_encoder.parameters())
    graph_param_ids = {id(p) for p in graph_params}
    other_params = [p for p in model.parameters() if id(p) not in graph_param_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": cfg.train.lr},
            {"params": graph_params, "lr": cfg.train.lr_graph},
        ],
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-6,
    )
    func_column_weights = torch.tensor(
        [FUNC_DEFAULT_WEIGHT_MAP.get(c, 1.0) for c in func_target_cols],
        dtype=torch.float32,
    )

    train_result = train_multitask(
        model=model,
        graph=train_graph,
        eval_graph=graph,
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
        main_logit_scale_learnable=cfg.train.main_logit_scale_learnable,
        main_logit_scale_min=cfg.train.main_logit_scale_min,
        main_logit_scale_max=cfg.train.main_logit_scale_max,
        domain_temperature=cfg.train.domain_temperature,
        main_loss_type=cfg.train.main_loss_type,
        aux_update_hgt=cfg.train.aux_update_hgt,
        aux_domain_interval=cfg.train.aux_domain_interval,
        aux_mvp_interval=cfg.train.aux_mvp_interval,
        aux_func_interval=cfg.train.aux_func_interval,
        func_loss_type=cfg.train.func_loss_type,
        func_smooth_l1_beta=cfg.train.func_smooth_l1_beta,
        gate_entropy_weight_start=cfg.train.gate_entropy_weight_start,
        gate_entropy_weight_end=cfg.train.gate_entropy_weight_end,
        func_column_scales=torch.tensor(func_target_scales, dtype=torch.float32),
        func_target_cols=func_target_cols,
        func_column_weights=func_column_weights,
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
