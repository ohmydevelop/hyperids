"""Tiny-GLiClass Student — 4L / 256D / 8H / FFN768 / context192.

    command → Tiny Security Encoder → command_emb
    score[i] = command_emb · label_emb[i] / sqrt(d)
    risk group (first 3) → softmax, rest (intent/tactic/technique) → sigmoid

Scoring dimension = 199 (schema_v1.all_label_ids() order), identical to the
Jev Teacher's soft-target vector, so KD is a straight vector-to-vector fit.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from schema_v1 import group_offsets

_OFF = group_offsets()
RISK_SLICE = slice(*_OFF["risk"])  # 0:3


class TinySecurityEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 768,
        max_len: int = 192,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L = input_ids.shape
        L = min(L, self.max_len)
        input_ids = input_ids[:, :L]
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)
        if attention_mask is None:
            mask = (input_ids != 0).float()
        else:
            mask = attention_mask[:, :L].float()
        src_key_padding_mask = (mask == 0.0)  # True = ignore
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        # mean pooling over non-pad tokens
        mask = mask.unsqueeze(-1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return x  # (B, d_model)


class TinyGLiClass(nn.Module):
    def __init__(
        self,
        num_labels: int = 199,
        vocab_size: int = 8000,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 768,
        max_len: int = 192,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.num_labels = num_labels
        self.encoder = TinySecurityEncoder(
            vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout
        )
        # learned label embeddings — one per schema label id
        self.label_emb = nn.Parameter(torch.empty(num_labels, d_model))
        nn.init.trunc_normal_(self.label_emb, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        cmd = self.encoder(input_ids, attention_mask)  # (B, d)
        cmd = F.normalize(cmd, dim=-1)
        lab = F.normalize(self.label_emb, dim=-1)
        logits = (cmd @ lab.T) / self.temperature  # (B, num_labels)
        return logits

    def logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """199-dim probability vector matching Jev soft-target layout."""
        risk = F.softmax(logits[:, RISK_SLICE], dim=-1)
        rest = torch.sigmoid(logits[:, 3:])
        return torch.cat([risk, rest], dim=-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    m = TinyGLiClass()
    ids = torch.randint(0, 8000, (2, 32))
    out = m(ids)
    print("logits:", tuple(out.shape))
    print("probs:", tuple(m.logits_to_probs(out).shape))
    print(f"params: {m.num_parameters()/1e6:.2f}M")
