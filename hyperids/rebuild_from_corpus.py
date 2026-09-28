"""从 corpus（Jev 199 维软标签，已 LFS 同步）一键重建 gliclass_v2 训练集。

链路：
  corpus/labeled/**/*.jsonl  →  合并 soft_labels_50k.jsonl
    →  prepare_data.build()  →  dataset/gliclass（199 标签 split）
    →  collapse.main()       →  dataset/gliclass_collapsed（132 split）
    →  build_dataset.main()  →  dataset/gliclass_v2（31 标签，最终训练集）

这是「从零复现」的训练数据起点（合成/打标等上游见 hyperids/data/，可选重跑）。
注意：corpus/labeled 是 soft_labels_50k 的超集（额外含 longtail/SUID/公开工具数据），
因此重建的训练集会比最初的 gliclass_v2 更全（80905 vs 75891 条）。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_LABELED = ROOT / "dataset" / "corpus" / "labeled"
SOFT_JSONL = ROOT / "dataset" / "soft_labels_50k.jsonl"


def merge_corpus_to_soft(corpus_dir: Path = CORPUS_LABELED, out: Path = SOFT_JSONL) -> int:
    """合并 corpus/labeled/**/*.jsonl → soft_labels_50k.jsonl，返回条数。"""
    rows = []
    seen = set()
    for p in sorted(corpus_dir.rglob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if not d.get("text") or "soft" not in d:
                continue
            if d["text"] in seen:
                continue
            seen.add(d["text"])
            rows.append(d)
    with open(out, "w", encoding="utf-8") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"merged {len(rows)} labeled rows -> {out}")
    return len(rows)


def main():
    merge_corpus_to_soft()

    from hyperids import prepare_data, collapse, build_dataset
    print("[1/3] prepare_data: 199 标签 → gliclass split ...")
    prepare_data.build()
    print("[2/3] collapse: 199 → 132 split ...")
    collapse.main()
    print("[3/3] build_dataset: 199 软标签 → 31 标签 gliclass_v2 ...")
    build_dataset.main()


if __name__ == "__main__":
    main()
