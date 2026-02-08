from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import HGTConv
except Exception as exc:  # pragma: no cover
    HGTConv = None
    _PYG_IMPORT_ERROR = exc
else:
    _PYG_IMPORT_ERROR = None


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphEncoder(nn.Module):
    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        gene_in_dim: int,
        trait_in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if _PYG_IMPORT_ERROR is not None:
            raise ImportError("torch_geometric is required for GraphEncoder") from _PYG_IMPORT_ERROR

        self.dropout = dropout
        self.input_proj = nn.ModuleDict(
            {
                "gene": nn.Linear(gene_in_dim, hidden_dim),
                "trait": nn.Linear(trait_in_dim, hidden_dim),
            }
        )
        self.input_norm = nn.ModuleDict(
            {
                "gene": nn.LayerNorm(hidden_dim),
                "trait": nn.LayerNorm(hidden_dim),
            }
        )

        self.convs = nn.ModuleList(
            [HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=num_heads) for _ in range(num_layers)]
        )
        self.out_proj = nn.ModuleDict(
            {
                "gene": nn.Linear(hidden_dim, out_dim),
                "trait": nn.Linear(hidden_dim, out_dim),
            }
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = {}
        for node_type, x in x_dict.items():
            h[node_type] = self.input_norm[node_type](self.input_proj[node_type](x))

        for conv in self.convs:
            h_new = conv(h, edge_index_dict)
            for node_type in h.keys():
                if node_type in h_new:
                    h[node_type] = F.dropout(
                        F.relu(h_new[node_type] + h[node_type]),
                        p=self.dropout,
                        training=self.training,
                    )

        gene_emb = self.out_proj["gene"](h["gene"])
        trait_emb = self.out_proj["trait"](h["trait"])
        return gene_emb, trait_emb


class TrilinearFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float,
        modality_drop_variant: float = 0.0,
        modality_drop_protein: float = 0.0,
        modality_drop_gene: float = 0.0,
    ) -> None:
        super().__init__()
        self.variant_proj = nn.Linear(dim, dim)
        self.protein_proj = nn.Linear(dim, dim)
        self.gene_proj = nn.Linear(dim, dim)
        self.variant_norm = nn.LayerNorm(dim)
        self.protein_norm = nn.LayerNorm(dim)
        self.gene_norm = nn.LayerNorm(dim)

        self.gate = nn.Sequential(
            nn.Linear(dim * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        self.final = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.modality_drop_variant = modality_drop_variant
        self.modality_drop_protein = modality_drop_protein
        self.modality_drop_gene = modality_drop_gene

        self.register_buffer("gate_bias", torch.tensor([0.1, 0.0, -0.1], dtype=torch.float32))

    def forward(
        self,
        variant_emb: torch.Tensor,
        protein_emb: torch.Tensor,
        gene_emb: torch.Tensor,
        gate_temperature: float = 1.0,
        return_gate_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            bsz = variant_emb.shape[0]
            mv = torch.ones(bsz, 1, device=variant_emb.device)
            mp = torch.ones(bsz, 1, device=variant_emb.device)
            mg = torch.ones(bsz, 1, device=variant_emb.device)

            if self.modality_drop_variant > 0:
                mv = torch.bernoulli(mv * (1.0 - self.modality_drop_variant))
            if self.modality_drop_protein > 0:
                mp = torch.bernoulli(mp * (1.0 - self.modality_drop_protein))
            if self.modality_drop_gene > 0:
                mg = torch.bernoulli(mg * (1.0 - self.modality_drop_gene))

            all_zero = (mv + mp + mg) == 0
            if all_zero.any():
                mv[all_zero] = 1.0

            variant_emb = variant_emb * mv
            protein_emb = protein_emb * mp
            gene_emb = gene_emb * mg

        v = self.variant_norm(self.variant_proj(variant_emb))
        p = self.protein_norm(self.protein_proj(protein_emb))
        g = self.gene_norm(self.gene_proj(gene_emb))

        logits = self.gate(torch.cat([v, p, g], dim=-1)) + self.gate_bias
        weights = F.softmax(logits / max(gate_temperature, 1e-3), dim=-1)

        fused = weights[:, 0:1] * variant_emb + weights[:, 1:2] * protein_emb + weights[:, 2:3] * gene_emb
        fused = self.final(fused)
        if return_gate_weights:
            return fused, weights
        return fused


class DiseaseEncoder(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
        )
        self.attn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.scale = math.sqrt(dim)

    def forward(
        self,
        trait_graph_emb: torch.Tensor,
        disease_ids: Sequence[int],
        disease_to_traits: Dict[int, List[int]],
    ) -> torch.Tensor:
        out: List[torch.Tensor] = []
        for disease_id in disease_ids:
            trait_ids = disease_to_traits.get(int(disease_id), [])
            if not trait_ids:
                # Keep behavior explicit to prevent silent leakage/degenerate disease vectors.
                raise ValueError(f"Disease {disease_id} has no mapped traits")

            x = trait_graph_emb[torch.tensor(trait_ids, device=trait_graph_emb.device)]  # [K, d]
            h = self.shared_mlp(x)  # [K, d]
            logits = self.attn(h) / self.scale  # [K, 1]
            weights = torch.softmax(logits, dim=0)  # [K, 1]
            pooled = (weights * h).sum(dim=0)  # [d]
            out.append(pooled)

        return torch.stack(out, dim=0)


class MVPRegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})")
        part = out_dim // num_heads
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, part),
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([h(x) for h in self.heads], dim=-1)


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        gene_in_dim: int,
        trait_in_dim: int,
        variant_in_dim: int,
        protein_in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_heads: int,
        num_graph_layers: int,
        dropout: float,
        num_domains: int,
        domain_embedding_dim: int,
        mvp_out_dim: int = 120,
        func_out_dim: int = 8,
        modality_drop_variant: float = 0.0,
        modality_drop_protein: float = 0.0,
        modality_drop_gene: float = 0.0,
        main_temperature: float = 0.15,
        main_logit_scale_learnable: bool = True,
    ) -> None:
        super().__init__()
        self.graph_encoder = GraphEncoder(
            metadata=metadata,
            gene_in_dim=gene_in_dim,
            trait_in_dim=trait_in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_graph_layers,
            dropout=dropout,
        )
        self.variant_encoder = MLPEncoder(variant_in_dim, hidden_dim, out_dim, dropout)
        self.protein_encoder = MLPEncoder(protein_in_dim, hidden_dim, out_dim, dropout)
        self.fusion = TrilinearFusion(
            out_dim,
            dropout,
            modality_drop_variant=modality_drop_variant,
            modality_drop_protein=modality_drop_protein,
            modality_drop_gene=modality_drop_gene,
        )

        self.disease_encoder = DiseaseEncoder(out_dim, hidden_dim, dropout)

        self.clip_variant_proj = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.clip_disease_proj = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        init_main_scale = 1.0 / max(main_temperature, 1e-6)
        self.main_logit_scale_log = nn.Parameter(
            torch.tensor(math.log(init_main_scale), dtype=torch.float32),
            requires_grad=main_logit_scale_learnable,
        )

        self.domain_variant_proj = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.domain_transform = nn.Linear(domain_embedding_dim, out_dim)
        self.num_domains = num_domains

        self.mvp_head = MVPRegressionHead(out_dim, hidden_dim, mvp_out_dim, num_heads=4, dropout=dropout)
        self.func_head = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, func_out_dim),
        )

    def forward_graph(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.graph_encoder(x_dict, edge_index_dict)

    def encode_variant(
        self,
        variant_ids: torch.Tensor,
        gene_ids: torch.Tensor,
        variant_x: torch.Tensor,
        protein_x: torch.Tensor,
        gene_graph_emb: torch.Tensor,
        gate_temperature: float = 1.0,
        return_gate_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        v = self.variant_encoder(variant_x.index_select(0, variant_ids))
        p = self.protein_encoder(protein_x.index_select(0, variant_ids))
        g = gene_graph_emb.index_select(0, gene_ids)
        return self.fusion(
            v,
            p,
            g,
            gate_temperature=gate_temperature,
            return_gate_weights=return_gate_weights,
        )

    def get_main_logit_scale(
        self,
        min_scale: float = 1.0,
        max_scale: float = 100.0,
    ) -> torch.Tensor:
        return torch.exp(self.main_logit_scale_log).clamp(min=min_scale, max=max_scale)

    def encode_disease_batch(
        self,
        disease_ids: Sequence[int],
        disease_to_traits: Dict[int, List[int]],
        trait_graph_emb: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.disease_encoder(trait_graph_emb, disease_ids, disease_to_traits)
        return F.normalize(self.clip_disease_proj(raw), dim=-1)

    def forward_main(
        self,
        variant_ids: torch.Tensor,
        gene_ids: torch.Tensor,
        variant_x: torch.Tensor,
        protein_x: torch.Tensor,
        gene_graph_emb: torch.Tensor,
        trait_graph_emb: torch.Tensor,
        disease_ids: Sequence[int],
        disease_to_traits: Dict[int, List[int]],
        gate_temperature: float = 1.0,
        return_gate_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_v_out = self.encode_variant(
            variant_ids,
            gene_ids,
            variant_x,
            protein_x,
            gene_graph_emb,
            gate_temperature=gate_temperature,
            return_gate_weights=return_gate_weights,
        )
        if return_gate_weights:
            z_v, gate_weights = z_v_out
        else:
            z_v = z_v_out
        z_v = F.normalize(self.clip_variant_proj(z_v), dim=-1)
        z_d = self.encode_disease_batch(disease_ids, disease_to_traits, trait_graph_emb)
        if return_gate_weights:
            return z_v, z_d, gate_weights
        return z_v, z_d

    def forward_domain(
        self,
        variant_ids: torch.Tensor,
        gene_ids: torch.Tensor,
        variant_x: torch.Tensor,
        protein_x: torch.Tensor,
        gene_graph_emb: torch.Tensor,
        domain_embeddings: torch.Tensor,
        temperature: float = 0.15,
        gate_temperature: float = 1.0,
        return_gate_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        z_v_out = self.encode_variant(
            variant_ids,
            gene_ids,
            variant_x,
            protein_x,
            gene_graph_emb,
            gate_temperature=gate_temperature,
            return_gate_weights=return_gate_weights,
        )
        if return_gate_weights:
            z_v, gate_weights = z_v_out
        else:
            z_v = z_v_out
        z_v = F.normalize(self.domain_variant_proj(z_v), dim=-1)
        z_p = F.normalize(self.domain_transform(domain_embeddings), dim=-1)
        logits = z_v @ z_p.t() / max(temperature, 1e-6)
        if return_gate_weights:
            return logits, gate_weights
        return logits

    def forward_mvp_reg(
        self,
        variant_ids: torch.Tensor,
        gene_ids: torch.Tensor,
        variant_x: torch.Tensor,
        protein_x: torch.Tensor,
        gene_graph_emb: torch.Tensor,
        gate_temperature: float = 1.0,
        return_gate_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        z_v_out = self.encode_variant(
            variant_ids,
            gene_ids,
            variant_x,
            protein_x,
            gene_graph_emb,
            gate_temperature=gate_temperature,
            return_gate_weights=return_gate_weights,
        )
        if return_gate_weights:
            z_v, gate_weights = z_v_out
        else:
            z_v = z_v_out
        out = self.mvp_head(z_v)
        if return_gate_weights:
            return out, gate_weights
        return out

    def forward_func(
        self,
        variant_ids: torch.Tensor,
        gene_ids: torch.Tensor,
        variant_x: torch.Tensor,
        protein_x: torch.Tensor,
        gene_graph_emb: torch.Tensor,
        gate_temperature: float = 1.0,
        return_gate_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        z_v_out = self.encode_variant(
            variant_ids,
            gene_ids,
            variant_x,
            protein_x,
            gene_graph_emb,
            gate_temperature=gate_temperature,
            return_gate_weights=return_gate_weights,
        )
        if return_gate_weights:
            z_v, gate_weights = z_v_out
        else:
            z_v = z_v_out
        out = self.func_head(z_v)
        if return_gate_weights:
            return out, gate_weights
        return out
