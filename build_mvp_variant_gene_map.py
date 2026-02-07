from __future__ import annotations

import argparse
from collections import defaultdict
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd


def normalize_id(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def parse_gene_tokens(raw: object) -> List[str]:
    text = normalize_id(raw)
    if not text:
        return []
    tokens = [t.strip() for t in re.split(r"[;|,]", text) if t.strip()]
    dedup: List[str] = []
    for token in tokens:
        if token not in dedup:
            dedup.append(token)
    return dedup


def pick_preferred_gene(tokens: List[str]) -> str:
    if not tokens:
        return ""
    non_ensembl = [t for t in tokens if not t.startswith("ensg")]
    if non_ensembl:
        return non_ensembl[0]
    return tokens[0]


def load_func_gene_map(func_labels: str) -> Dict[str, Set[str]]:
    df = pd.read_csv(func_labels, usecols=["snp", "gene_id"])
    out: Dict[str, Set[str]] = defaultdict(set)
    for snp, raw_gene in df.itertuples(index=False):
        rsid = normalize_id(snp)
        if not rsid:
            continue
        tokens = parse_gene_tokens(raw_gene)
        gene = pick_preferred_gene(tokens)
        if gene:
            out[rsid].add(gene)
    return out


def load_hgvs_gene_map(
    hgvs_embed_csv: str,
    rsid_embed_csv: str,
    chunksize: int = 200000,
) -> Dict[str, Set[str]]:
    pat = re.compile(r"\(([^)]+)\)")
    out: Dict[str, Set[str]] = defaultdict(set)

    hgvs_iter = pd.read_csv(hgvs_embed_csv, usecols=["variant_id"], chunksize=chunksize)
    rsid_iter = pd.read_csv(rsid_embed_csv, usecols=["variant_id"], chunksize=chunksize)

    for hgvs_chunk, rsid_chunk in zip(hgvs_iter, rsid_iter):
        if len(hgvs_chunk) != len(rsid_chunk):
            raise ValueError("HGVS and RSID embedding files have mismatched row counts")

        hgvs_ids = hgvs_chunk["variant_id"].astype(str).tolist()
        rsids = rsid_chunk["variant_id"].map(normalize_id).tolist()

        for hgvs_id, rsid in zip(hgvs_ids, rsids):
            if not rsid:
                continue
            m = pat.search(hgvs_id)
            if not m:
                continue
            gene = normalize_id(m.group(1))
            if gene:
                out[rsid].add(gene)

    return out


def build_mapping(
    clinvar_variant_csv: str,
    func_labels_csv: str,
    hgvs_embed_csv: str,
    rsid_embed_csv: str,
) -> pd.DataFrame:
    mvp_df = pd.read_csv(clinvar_variant_csv, usecols=["variant"]) 
    mvp_variants = sorted({normalize_id(v) for v in mvp_df["variant"].tolist() if normalize_id(v)})

    func_map = load_func_gene_map(func_labels_csv)
    hgvs_map = load_hgvs_gene_map(hgvs_embed_csv, rsid_embed_csv)

    rows = []
    for rsid in mvp_variants:
        hgvs_genes = sorted(hgvs_map.get(rsid, set()))
        func_genes = sorted(func_map.get(rsid, set()))

        if hgvs_genes and func_genes:
            overlap = sorted(set(hgvs_genes) & set(func_genes))
            if overlap:
                gene_id = overlap[0]
                source = "hgvs_func_agree"
                conflict = 0
            else:
                gene_id = hgvs_genes[0]
                source = "hgvs_preferred_conflict"
                conflict = 1
        elif hgvs_genes:
            gene_id = hgvs_genes[0]
            source = "hgvs_only"
            conflict = 0
        elif func_genes:
            gene_id = func_genes[0]
            source = "func_only"
            conflict = 0
        else:
            gene_id = ""
            source = "unresolved"
            conflict = 0

        rows.append(
            {
                "variant_id": rsid,
                "gene_id": gene_id,
                "source": source,
                "conflict_flag": conflict,
                "gene_candidates_hgvs": "|".join(hgvs_genes),
                "gene_candidates_func": "|".join(func_genes),
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MVP variant->gene_id mapping file")
    parser.add_argument(
        "--mvp-variants",
        default="../data/MVP_data/Variant_phenotype_correlation_matrix/ClinVar_variants_all_with_ref_EUR.csv",
    )
    parser.add_argument(
        "--func-labels",
        default="../data/func_impact_data/processed/func_impact_labels.csv",
    )
    parser.add_argument(
        "--hgvs-embeddings",
        default="../data/MVP_data/embeddings/variant_x.csv",
    )
    parser.add_argument(
        "--rsid-embeddings",
        default="../data/MVP_data/embeddings/variant_x_rsid.csv",
    )
    parser.add_argument(
        "--output",
        default="artifacts/mvp_variant_gene_id_map.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_df = build_mapping(
        clinvar_variant_csv=args.mvp_variants,
        func_labels_csv=args.func_labels,
        hgvs_embed_csv=args.hgvs_embeddings,
        rsid_embed_csv=args.rsid_embeddings,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    total = len(out_df)
    resolved = int((out_df["gene_id"] != "").sum())
    unresolved = total - resolved
    conflicts = int((out_df["conflict_flag"] == 1).sum())

    print(f"output={out_path}")
    print(f"total_variants={total}")
    print(f"resolved={resolved}")
    print(f"unresolved={unresolved}")
    print(f"conflicts={conflicts}")
    print("source_counts=" + str(out_df["source"].value_counts().to_dict()))


if __name__ == "__main__":
    main()
