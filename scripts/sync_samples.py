"""生成 corpus 样本清单（dataset/MANIFEST.md）。

corpus 是 31 标签的最终数据源（Git LFS 同步）。本脚本仅扫描 corpus 生成清单，
不做拆分（上游合成/打标见 hyperids.data，可选重跑）。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "dataset"
CORPUS = DS / "corpus"


def count(path: Path) -> int:
    return sum(1 for l in path.read_text(encoding="utf-8").splitlines() if l.strip())


def main():
    lines = ["# 样本清单（Sample Manifest）", "",
             "> 分类维度：`来源(source) × 标注(raw/labeled)`。",
             "> `corpus/` 为 git 同步的正式语料（31 标签：verdict 3 + action 28）。",
             "", "## 标注", "- **raw**：原始样本，未打标签。", "- **labeled**：Jev 打过标签（verdict + actions + soft）。",
             "", "## 来源", "- **public**：`quasarnix` / `nl2bash` / `benign` / `suspicious` / `gtfobins` / `payloads` / `curated`。",
             "- **synthetic**：`synthetic` / `diverse` / `suid_variants` / `longtail`。",
             "", "## 文件清单"]
    for p in sorted(CORPUS.rglob("*.jsonl")):
        lines.append(f"- `{p.relative_to(DS)}`  ({p.stat().st_size/1e6:.2f}MB, {count(p)} 条)")
    (DS / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote dataset/MANIFEST.md")


if __name__ == "__main__":
    main()
