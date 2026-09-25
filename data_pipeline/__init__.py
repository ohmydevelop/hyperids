"""Frontier LLM data pipeline: label, synthesize, obfuscate, hard-negative, minimal-pair.

Stages (all produce rows in the same schema):
  1. seed        — load real benign/malicious command seeds
  2. label       — Frontier LLM assigns risk/intent/tactic/technique
  3. synthesize  — LLM generates new synthetic commands per label combo
  4. obfuscate   — LLM creates obfuscated variants of known samples
  5. hard_neg    — LLM generates near-miss / confusing negatives
  6. min_pair    — LLM creates minimal pairs (benign↔malicious single-edit)
  7. export      — write train/val/test parquet shards

Output row schema:
  {
    "text": str,                  # raw command / script
    "labels": list[str],          # subset of schema.all_label_ids()
    "source": str,                # seed|synthetic|obfuscated|hard_neg|min_pair
    "parent": str | None,         # id of source row for derived samples
    "meta": dict,                 # free-form (model, temperature, prompt hash)
  }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sample:
    text: str
    labels: list[str]
    source: str
    parent: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = ""
