"""从 corpus（31 维 Jev 软标签，已 LFS 同步）一键重建 gliclass_v2 训练集。

链路：
  corpus/labeled/**/*.jsonl  →  build_dataset.build()  →  dataset/gliclass_v2

这是「从零复现」的训练数据起点（合成/打标等上游见 hyperids/data/，可选重跑）。
"""
from __future__ import annotations

from hyperids import build_dataset


def main():
    build_dataset.build()


if __name__ == "__main__":
    main()
