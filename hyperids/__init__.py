"""HyperIDs — 恶意 shell 命令威胁分类（GLiClass + bert-small，31 标签）。

包结构：
  hyperids.schema             当前 31 标签 schema + MITRE 规则推导
  hyperids.schema_legacy      199 标签 schema（仅供 199→31 映射）
  hyperids.llm / jev_client   公共客户端（LLM 网关 / Jev SystemOne）
  hyperids.train/eval/predict 训练/评估/推理（当前 v4）
  hyperids.build_dataset      199 软标签 → 31 标签训练集
  hyperids.collapse           199 → 132 折叠（提供 train/val/test split）
  hyperids.data               上游数据准备（拉取/合成/打标，可选重跑）
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
