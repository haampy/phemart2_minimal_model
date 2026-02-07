from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.transforms import ToUndirected
except Exception as exc:  # pragma: no cover
    HeteroData = Any
    ToUndirected = None
    _PYG_IMPORT_ERROR = exc
else:
    _PYG_IMPORT_ERROR = None


FUNC_TARGET_COLS = [
    "CADD_phred",
    "phyloP",
    "GERP++",
    "SIFT",
    "Polyphen2_HDIV",
    "MetaSVM",
    "REVEL",
    "AlphaMissense",
]

FUNC_MASK_COLS = [f"{c}_mask" for c in FUNC_TARGET_COLS]


def normalize_id(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def parse_hpo_ids(raw: Any) -> List[str]:
    if pd.isna(raw):
        return []
    if isinstance(raw, (list, tuple)):
        return [normalize_id(v) for v in raw if normalize_id(v)]
    text = str(raw)
    return [normalize_id(v) for v in text.split("|") if normalize_id(v)]


def _first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _standardize_variant_gene_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    var_col = _first_existing(df, ["variant_id", "variant", "snps", "snp", "variant_raw"])
    gene_col = _first_existing(df, ["gene_id", "gene_name", "genes", "gene"])
    if var_col is None:
        raise ValueError("Cannot find variant column")
    if gene_col is None:
        raise ValueError("Cannot find gene column")
    if var_col != "variant_id":
        df = df.rename(columns={var_col: "variant_id"})
    if gene_col != "gene_id":
        df = df.rename(columns={gene_col: "gene_id"})
    df["variant_id"] = df["variant_id"].map(normalize_id)
    df["gene_id"] = df["gene_id"].map(normalize_id)
    df = df[(df["variant_id"] != "") & (df["gene_id"] != "")]
    return df


def load_main_labels(path: str) -> pd.DataFrame:
    df = _standardize_variant_gene_columns(_read_csv(path))
    disease_col = _first_existing(df, ["disease_index", "disease_id"])
    hpo_col = _first_existing(df, ["hpo_ids", "hpo_id"])
    if disease_col is None:
        raise ValueError("Cannot find disease_index column in main labels")
    if disease_col != "disease_index":
        df = df.rename(columns={disease_col: "disease_index"})
    if hpo_col and hpo_col != "hpo_ids":
        df = df.rename(columns={hpo_col: "hpo_ids"})
    if "hpo_ids" not in df.columns:
        df["hpo_ids"] = ""
    df = df.dropna(subset=["disease_index"])
    df["disease_index"] = df["disease_index"].astype(int)
    return df[["variant_id", "gene_id", "disease_index", "hpo_ids"]]


def load_disease_table(path: str) -> pd.DataFrame:
    df = _read_csv(path)
    disease_col = _first_existing(df, ["disease_index", "disease_id"])
    hpo_col = _first_existing(df, ["hpo_ids", "hpo_id"])
    if disease_col is None or hpo_col is None:
        raise ValueError("Disease table must contain disease_index and hpo_ids")
    df = df.rename(columns={disease_col: "disease_index", hpo_col: "hpo_ids"})
    df = df.dropna(subset=["disease_index"])
    df["disease_index"] = df["disease_index"].astype(int)
    return df[["disease_index", "hpo_ids"]].drop_duplicates("disease_index")


def load_domain_labels(path: str) -> pd.DataFrame:
    df = _standardize_variant_gene_columns(_read_csv(path))
    domain_col = _first_existing(df, ["domain_map", "domain", "label"])
    if domain_col is None:
        raise ValueError("Cannot find domain label column")
    if domain_col != "domain_map":
        df = df.rename(columns={domain_col: "domain_map"})
    df = df.dropna(subset=["domain_map"])
    df["domain_map"] = df["domain_map"].astype(int)
    return df[["variant_id", "gene_id", "domain_map"]]


def load_func_labels(path: str) -> pd.DataFrame:
    df = _standardize_variant_gene_columns(_read_csv(path))
    missing_targets = [c for c in FUNC_TARGET_COLS if c not in df.columns]
    missing_masks = [c for c in FUNC_MASK_COLS if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing FUNC target columns: {missing_targets}")
    if missing_masks:
        raise ValueError(f"Missing FUNC mask columns: {missing_masks}")
    cols = ["variant_id", "gene_id"] + FUNC_TARGET_COLS + FUNC_MASK_COLS
    df = df[cols].dropna(subset=["variant_id", "gene_id"])
    for c in FUNC_TARGET_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in FUNC_MASK_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float32)
    return df


def load_embeddings(
    path: str,
    required_ids: Optional[Set[str]] = None,
    index_col: Optional[str] = None,
    chunksize: int = 50000,
) -> pd.DataFrame:
    if required_ids is None:
        df = pd.read_csv(path, index_col=0)
        df.index = df.index.map(normalize_id)
        df = df[~df.index.duplicated(keep="first")]
        return df.astype(np.float32)

    collected: List[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        idx_col = index_col or chunk.columns[0]
        chunk = chunk.rename(columns={idx_col: "__id__"})
        chunk["__id__"] = chunk["__id__"].map(normalize_id)
        sub = chunk[chunk["__id__"].isin(required_ids)]
        if not sub.empty:
            collected.append(sub)

    if not collected:
        return pd.DataFrame(dtype=np.float32)

    df = pd.concat(collected, axis=0)
    df = df.drop_duplicates(subset=["__id__"], keep="first").set_index("__id__")
    return df.astype(np.float32)


def load_mvp_reg_targets(variant_map_csv: str, target_npy: str) -> pd.DataFrame:
    map_df = pd.read_csv(variant_map_csv)
    variant_col = _first_existing(map_df, ["variant", "variant_id", "snp", "variant_raw"])
    gene_col = _first_existing(map_df, ["gene_id", "gene", "gene_name", "genes"])
    if variant_col is None:
        raise ValueError("Cannot find MVP variant id column")

    targets = np.load(target_npy)
    if len(map_df) != len(targets):
        raise ValueError(
            f"MVP mapping rows ({len(map_df)}) != target rows ({len(targets)})"
        )

    out = pd.DataFrame({
        "variant_id": map_df[variant_col].map(normalize_id),
        "gene_id": (
            map_df[gene_col].map(normalize_id)
            if gene_col is not None
            else [""] * len(map_df)
        ),
    })
    targets = targets.astype(np.float32)
    norms = np.linalg.norm(targets, axis=1, keepdims=True)
    targets = targets / np.clip(norms, 1e-12, None)

    target_cols = [f"mvp_{i}" for i in range(targets.shape[1])]
    target_df = pd.DataFrame(targets, columns=target_cols)
    out = pd.concat([out.reset_index(drop=True), target_df], axis=1)

    out = out[out["variant_id"] != ""]
    out = out.drop_duplicates(subset=["variant_id"], keep="first")
    return out


def _split_items(
    items: Sequence[str],
    seed: int,
    ratios: Tuple[float, float, float],
) -> Dict[str, str]:
    train_r, val_r, test_r = ratios
    total = train_r + val_r + test_r
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    items = list(set(items))
    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_r)
    n_val = int(n * val_r)

    mapping: Dict[str, str] = {}
    for item in items[:n_train]:
        mapping[item] = "train"
    for item in items[n_train : n_train + n_val]:
        mapping[item] = "val"
    for item in items[n_train + n_val :]:
        mapping[item] = "test"
    return mapping


def build_global_variant_split(
    main_df: pd.DataFrame,
    domain_df: pd.DataFrame,
    mvp_df: pd.DataFrame,
    func_df: pd.DataFrame,
    seed: int,
    ratios: Tuple[float, float, float],
) -> Dict[str, str]:
    dfs = [main_df, domain_df, mvp_df, func_df]
    for df in dfs:
        if "variant_id" not in df.columns or "gene_id" not in df.columns:
            raise ValueError("All task dataframes must contain variant_id and gene_id")

    main_genes = [g for g in main_df["gene_id"].dropna().unique().tolist() if g]
    gene_to_split = _split_items(main_genes, seed, ratios)

    for i, aux_df in enumerate([domain_df, mvp_df, func_df], start=1):
        aux_genes = [g for g in aux_df["gene_id"].dropna().unique().tolist() if g]
        new_genes = [g for g in aux_genes if g not in gene_to_split]
        if new_genes:
            aux_split = _split_items(new_genes, seed + i, ratios)
            gene_to_split.update(aux_split)

    variant_to_split: Dict[str, str] = {}
    unresolved_variants: Set[str] = set()
    conflict_variants: Set[str] = set()
    for df in dfs:
        for row in df[["variant_id", "gene_id"]].itertuples(index=False):
            v, g = row
            if not v:
                continue
            split = gene_to_split.get(g) if g else None
            if split is None:
                unresolved_variants.add(v)
                continue
            prev = variant_to_split.get(v)
            if prev is not None and prev != split:
                conflict_variants.add(v)
                continue
            variant_to_split[v] = split

    for v in conflict_variants:
        variant_to_split.pop(v, None)

    # Variant-level fallback for rows without gene assignment.
    unresolved_variants = unresolved_variants - set(variant_to_split.keys())
    unresolved_variants = unresolved_variants - conflict_variants
    if unresolved_variants:
        variant_split = _split_items(sorted(unresolved_variants), seed + 99, ratios)
        variant_to_split.update(variant_split)

    return variant_to_split


def apply_split(
    df: pd.DataFrame,
    variant_col_or_index: str,
    split_map: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    if variant_col_or_index == "index":
        variants = work.index.map(normalize_id)
    else:
        variants = work[variant_col_or_index].map(normalize_id)
    work["_split"] = variants.map(split_map)
    work = work.dropna(subset=["_split"])

    train_df = work[work["_split"] == "train"].drop(columns=["_split"])
    val_df = work[work["_split"] == "val"].drop(columns=["_split"])
    test_df = work[work["_split"] == "test"].drop(columns=["_split"])
    return train_df, val_df, test_df


def build_mappings(
    gene_x_df: pd.DataFrame,
    trait_x_df: pd.DataFrame,
    disease_df: pd.DataFrame,
) -> Dict[str, Dict[Any, Any]]:
    gene_ids = [normalize_id(v) for v in gene_x_df.index.tolist()]
    trait_ids = [normalize_id(v) for v in trait_x_df.index.tolist()]

    gene_to_idx = {g: i for i, g in enumerate(gene_ids)}
    trait_to_idx = {t: i for i, t in enumerate(trait_ids)}

    disease_ids = sorted(disease_df["disease_index"].astype(int).unique().tolist())
    disease_to_idx = {d: i for i, d in enumerate(disease_ids)}

    return {
        "gene_to_idx": gene_to_idx,
        "idx_to_gene": {i: g for g, i in gene_to_idx.items()},
        "trait_to_idx": trait_to_idx,
        "idx_to_trait": {i: t for t, i in trait_to_idx.items()},
        "disease_to_idx": disease_to_idx,
        "idx_to_disease": {i: d for d, i in disease_to_idx.items()},
    }


def _edges_from_file(
    edge_path: str,
    src_mapping: Dict[str, int],
    dst_mapping: Dict[str, int],
) -> torch.Tensor:
    df = pd.read_csv(edge_path)
    if len(df.columns) < 2:
        raise ValueError(f"Edge file {edge_path} must have at least 2 columns")
    c0, c1 = df.columns[:2]
    src = df[c0].map(normalize_id)
    dst = df[c1].map(normalize_id)

    src_idx = src.map(src_mapping)
    dst_idx = dst.map(dst_mapping)
    valid = src_idx.notna() & dst_idx.notna()

    src_vals = src_idx[valid].astype(np.int64).to_numpy()
    dst_vals = dst_idx[valid].astype(np.int64).to_numpy()
    if len(src_vals) == 0:
        raise ValueError(f"No valid edges in {edge_path}")

    return torch.tensor(np.stack([src_vals, dst_vals], axis=0), dtype=torch.long)


def build_hetero_graph(
    gene_x_df: pd.DataFrame,
    trait_x_df: pd.DataFrame,
    edge_files: Dict[str, str],
    gene_mapping: Dict[str, int],
    trait_mapping: Dict[str, int],
) -> HeteroData:
    if _PYG_IMPORT_ERROR is not None:
        raise ImportError(
            "torch_geometric is required for graph construction"
        ) from _PYG_IMPORT_ERROR

    data = HeteroData()
    data["gene"].x = torch.tensor(gene_x_df.to_numpy(dtype=np.float32), dtype=torch.float32)
    data["trait"].x = torch.tensor(
        trait_x_df.to_numpy(dtype=np.float32), dtype=torch.float32
    )

    data[("gene", "to", "gene")].edge_index = _edges_from_file(
        edge_files["gene_to_gene"], gene_mapping, gene_mapping
    )
    data[("gene", "to", "trait")].edge_index = _edges_from_file(
        edge_files["gene_to_trait"], gene_mapping, trait_mapping
    )
    data[("trait", "to", "trait")].edge_index = _edges_from_file(
        edge_files["trait_to_trait"], trait_mapping, trait_mapping
    )

    data = ToUndirected()(data)
    return data


def build_disease_to_traits_map(
    disease_df: pd.DataFrame,
    trait_mapping: Dict[str, int],
) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for row in disease_df[["disease_index", "hpo_ids"]].itertuples(index=False):
        disease_id, raw = int(row[0]), row[1]
        trait_ids = [trait_mapping[t] for t in parse_hpo_ids(raw) if t in trait_mapping]
        if trait_ids:
            out[disease_id] = sorted(set(trait_ids))
    return out


def build_variant_positive_map(
    main_df: pd.DataFrame,
    variant_mapping: Dict[str, int],
) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    for row in main_df[["variant_id", "disease_index"]].itertuples(index=False):
        variant_id, disease_id = row
        idx = variant_mapping.get(variant_id)
        if idx is None:
            continue
        out.setdefault(idx, set()).add(int(disease_id))
    return out


@dataclass
class FeatureStore:
    variant_to_idx: Dict[str, int]
    idx_to_variant: Dict[int, str]
    variant_x: torch.Tensor
    protein_x: torch.Tensor


def build_feature_store(
    variant_x_df: pd.DataFrame,
    protein_x_df: pd.DataFrame,
) -> FeatureStore:
    shared_ids = sorted(set(variant_x_df.index.tolist()) & set(protein_x_df.index.tolist()))
    if not shared_ids:
        raise ValueError("No overlapping variant ids between variant_x and protein_x")

    v_df = variant_x_df.loc[shared_ids]
    p_df = protein_x_df.loc[shared_ids]

    variant_to_idx = {v: i for i, v in enumerate(shared_ids)}
    idx_to_variant = {i: v for v, i in variant_to_idx.items()}
    variant_x = torch.tensor(v_df.to_numpy(dtype=np.float32), dtype=torch.float32)
    protein_x = torch.tensor(p_df.to_numpy(dtype=np.float32), dtype=torch.float32)

    return FeatureStore(
        variant_to_idx=variant_to_idx,
        idx_to_variant=idx_to_variant,
        variant_x=variant_x,
        protein_x=protein_x,
    )


class DictDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


def make_main_records(
    main_df: pd.DataFrame,
    variant_to_idx: Dict[str, int],
    gene_to_idx: Dict[str, int],
) -> List[Dict[str, Any]]:
    grouped = (
        main_df.groupby(["variant_id", "gene_id"], as_index=False)["disease_index"]
        .apply(list)
        .reset_index(drop=True)
    )

    records: List[Dict[str, Any]] = []
    for row in grouped.itertuples(index=False):
        variant_id, gene_id, disease_ids = row
        v_idx = variant_to_idx.get(variant_id)
        g_idx = gene_to_idx.get(gene_id)
        if v_idx is None or g_idx is None:
            continue
        pos = sorted(set(int(d) for d in disease_ids))
        if not pos:
            continue
        records.append(
            {
                "variant_idx": v_idx,
                "gene_idx": g_idx,
                "positive_disease_ids": pos,
            }
        )
    return records


def make_domain_records(
    df: pd.DataFrame,
    variant_to_idx: Dict[str, int],
    gene_to_idx: Dict[str, int],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in df[["variant_id", "gene_id", "domain_map"]].itertuples(index=False):
        v, g, y = row
        v_idx = variant_to_idx.get(v)
        g_idx = gene_to_idx.get(g)
        if v_idx is None or g_idx is None:
            continue
        records.append({"variant_idx": v_idx, "gene_idx": g_idx, "label": int(y)})
    return records


def make_mvp_records(
    df: pd.DataFrame,
    variant_to_idx: Dict[str, int],
    gene_to_idx: Dict[str, int],
    target_cols: Sequence[str],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    cols = ["variant_id", "gene_id"] + list(target_cols)
    for row in df[cols].itertuples(index=False):
        variant_id = row[0]
        gene_id = row[1]
        target = np.asarray(row[2:], dtype=np.float32)
        v_idx = variant_to_idx.get(variant_id)
        g_idx = gene_to_idx.get(gene_id)
        if v_idx is None or g_idx is None:
            continue
        records.append(
            {
                "variant_idx": v_idx,
                "gene_idx": g_idx,
                "target": target,
            }
        )
    return records


def make_func_records(
    df: pd.DataFrame,
    variant_to_idx: Dict[str, int],
    gene_to_idx: Dict[str, int],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    cols = ["variant_id", "gene_id"] + FUNC_TARGET_COLS + FUNC_MASK_COLS
    for row in df[cols].itertuples(index=False):
        variant_id = row[0]
        gene_id = row[1]
        target = np.asarray(row[2 : 2 + len(FUNC_TARGET_COLS)], dtype=np.float32)
        mask = np.asarray(row[2 + len(FUNC_TARGET_COLS) :], dtype=np.float32)
        v_idx = variant_to_idx.get(variant_id)
        g_idx = gene_to_idx.get(gene_id)
        if v_idx is None or g_idx is None:
            continue
        records.append(
            {
                "variant_idx": v_idx,
                "gene_idx": g_idx,
                "target": target,
                "mask": mask,
            }
        )
    return records


def _collate_main(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "variant_idx": torch.tensor([b["variant_idx"] for b in batch], dtype=torch.long),
        "gene_idx": torch.tensor([b["gene_idx"] for b in batch], dtype=torch.long),
        "positive_disease_ids": [b["positive_disease_ids"] for b in batch],
    }


def _collate_classification(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "variant_idx": torch.tensor([b["variant_idx"] for b in batch], dtype=torch.long),
        "gene_idx": torch.tensor([b["gene_idx"] for b in batch], dtype=torch.long),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def _collate_regression(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "variant_idx": torch.tensor([b["variant_idx"] for b in batch], dtype=torch.long),
        "gene_idx": torch.tensor([b["gene_idx"] for b in batch], dtype=torch.long),
        "target": torch.tensor(np.stack([b["target"] for b in batch]), dtype=torch.float32),
    }


def _collate_func(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "variant_idx": torch.tensor([b["variant_idx"] for b in batch], dtype=torch.long),
        "gene_idx": torch.tensor([b["gene_idx"] for b in batch], dtype=torch.long),
        "target": torch.tensor(np.stack([b["target"] for b in batch]), dtype=torch.float32),
        "mask": torch.tensor(np.stack([b["mask"] for b in batch]), dtype=torch.float32),
    }


def make_dataloader_for_task(
    task_name: str,
    records: List[Dict[str, Any]],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = DictDataset(records)
    if task_name == "main":
        collate_fn = _collate_main
    elif task_name == "domain":
        collate_fn = _collate_classification
    elif task_name == "mvp":
        collate_fn = _collate_regression
    elif task_name == "func":
        collate_fn = _collate_func
    else:
        raise ValueError(f"Unknown task_name: {task_name}")

    effective_shuffle = shuffle and len(dataset) > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )


def summarize_split(
    split_map: Dict[str, str],
    task_dfs: Dict[str, pd.DataFrame],
) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for task_name, df in task_dfs.items():
        variants = set(df["variant_id"].tolist())
        counts = {"train": 0, "val": 0, "test": 0}
        for v in variants:
            split = split_map.get(v)
            if split in counts:
                counts[split] += 1
        summary[task_name] = counts
    return summary


def compute_train_test_overlap(
    task_splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
) -> Dict[str, int]:
    train_union: Set[str] = set()
    test_union: Set[str] = set()
    for train_df, _, test_df in task_splits.values():
        train_union.update(train_df["variant_id"].tolist())
        test_union.update(test_df["variant_id"].tolist())

    return {
        "train_variants": len(train_union),
        "test_variants": len(test_union),
        "train_test_overlap": len(train_union & test_union),
    }
