from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import torch


@dataclass
class PathsConfig:
    main_labels: str = "../data/main_task/output/expanded_labels.csv"
    disease_table: str = "../data/main_task/output/new_disease_to_traits.csv"
    variant_x: str = "../data/variant_data/output/variant_x.csv"
    protein_x: str = "../data/variant_data/output/gene_local_x_mean.csv"
    gene_x: str = "../data/gene_data/output/gene_global_x.csv"
    trait_x: str = "../data/trait_data/output/trait_x.csv"
    gene_to_gene: str = "../data/graph_data/output/gene_to_gene.csv"
    gene_to_trait: str = "../data/graph_data/output/gene_to_trait.csv"
    trait_to_trait: str = "../data/graph_data/output/trait_to_trait.csv"
    domain_labels: str = "../data/domain_data/processed/domain_labels.csv"
    domain_embeddings: str = "../data/domain_data/processed/domain_embeddings.csv"
    mvp_variant_map: str = (
        "../data/MVP_data/Variant_phenotype_correlation_matrix/"
        "ClinVar_variants_all_with_ref_EUR.csv"
    )
    mvp_variant_gene_map: str = "artifacts/mvp_variant_gene_id_map.csv"
    mvp_targets: str = (
        "../data/MVP_data/Variant_phenotype_correlation_matrix/"
        "ClinVar_EUR_variant_embedding_dim120_svd.npy"
    )
    mvp_hgvs_embeddings: str = "../data/MVP_data/embeddings/variant_x.csv"
    mvp_rsid_embeddings: str = "../data/MVP_data/embeddings/variant_x_rsid.csv"
    func_labels: str = "../data/func_impact_data/processed/func_impact_labels.csv"
    output_dir: str = "experiments/minimal_multitask"


@dataclass
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42


@dataclass
class ModelConfig:
    variant_in_dim: int = 1280
    protein_in_dim: int = 1280
    graph_gene_in_dim: int = 768
    graph_trait_in_dim: int = 768
    hidden_dim: int = 256
    out_dim: int = 128
    num_heads: int = 2
    num_graph_layers: int = 2
    dropout: float = 0.3
    num_domains: int = 769
    domain_embedding_dim: int = 768
    func_out_dim: int = 8
    mvp_out_dim: int = 120
    modality_drop_variant: float = 0.10
    modality_drop_protein: float = 0.10
    modality_drop_gene: float = 0.20


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size_main: int = 128
    batch_size_domain: int = 128
    batch_size_mvp: int = 128
    batch_size_func: int = 128
    lr: float = 1e-4
    lr_graph: float = 3e-5
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 10
    main_temperature: float = 0.15
    main_logit_scale_learnable: bool = True
    main_logit_scale_min: float = 1.0
    main_logit_scale_max: float = 100.0
    domain_temperature: float = 0.15
    main_loss_type: str = "softmax"  # {"softmax", "bce"}
    aux_update_hgt: bool = False
    aux_domain_interval: int = 1
    aux_mvp_interval: int = 1
    aux_func_interval: int = 1
    use_inductive_graph_train: bool = False
    func_loss_type: str = "smooth_l1"  # {"smooth_l1", "mse"}
    func_smooth_l1_beta: float = 1.0
    gate_entropy_weight_start: float = 1e-3
    gate_entropy_weight_end: float = 0.0
    func_active_scores: Tuple[str, ...] = ("CADD_phred", "phyloP", "GERP++", "MetaSVM")
    func_min_valid_scores: int = 4
    func_train_per_gene_cap: int = 50
    min_train_records_mvp: int = 1000
    min_train_records_func: int = 1000


@dataclass
class LossWeights:
    main: float = 1.0
    domain: float = 0.4
    mvp_reg: float = 0.05
    func: float = 0.05


@dataclass
class RuntimeConfig:
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    num_workers: int = 0


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def default_config() -> Config:
    return Config()


def ensure_output_dir(cfg: Config) -> Path:
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
