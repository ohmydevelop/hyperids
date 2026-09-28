"""Organize dataset samples into a corpus by (source) x (raw/labeled).

Naming follows the data-source convention (quasarnix / nl2bash / cowrie / ...),
not implementation details like "soft_labels_50k".

Output tree (plain JSONL, no gzip):
  dataset/corpus/
    raw/{public,synthetic}/<source>.jsonl      # unlabeled originals
    labeled/{public,synthetic}/<source>.jsonl  # Jev-labeled
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "dataset"
CORPUS = DS / "corpus"

PUBLIC_SRC = {"quasarnix", "nl2bash", "benign", "suspicious"}
SYNTH_SRC = {"synthetic", "diverse"}


def src_class(s):
    if s in PUBLIC_SRC: return "public"
    if s in SYNTH_SRC: return "synthetic"
    return "other"


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def split_by_source(src: Path, dst: Path):
    """Split a source-tagged jsonl into <class>/<source>.jsonl. Returns sizes."""
    buckets = defaultdict(list)
    for d in read_jsonl(src):
        buckets[(src_class(d.get("source", "?")), d.get("source", "?"))].append(d)
    for (cls, sname), rows in buckets.items():
        write_jsonl(dst / cls / f"{sname}.jsonl", rows)
    return {k: len(v) for k, v in buckets.items()}


def main():
    # ---- raw (unlabeled) ----
    split_by_source(DS / "candidates_50k.jsonl", CORPUS / "raw")
    # public tool candidates (GTFOBins / PayloadsAllTheThings)
    write_jsonl(CORPUS / "raw" / "public" / "gtfobins_payloads.jsonl",
                read_jsonl(DS / "public_candidates.jsonl"))
    # SUID variants (rule-generated)
    write_jsonl(CORPUS / "raw" / "synthetic" / "suid_variants.jsonl",
                read_jsonl(DS / "suid_variants.jsonl"))

    # ---- labeled (Jev) ----
    split_by_source(DS / "soft_labels_50k.jsonl", CORPUS / "labeled")
    write_jsonl(CORPUS / "labeled" / "public" / "gtfobins_payloads.jsonl",
                read_jsonl(DS / "public_labeled.jsonl"))
    # longtail synth (merge the three synth_longtail files)
    longtail = []
    for name in ["synth_longtail", "synth_longtail_bulk", "synth_longtail_quality"]:
        p = DS / f"{name}.jsonl"
        if p.exists():
            longtail += read_jsonl(p)
    write_jsonl(CORPUS / "labeled" / "synthetic" / "longtail.jsonl", longtail)
    # SUID labeled
    write_jsonl(CORPUS / "labeled" / "synthetic" / "suid_variants.jsonl",
                read_jsonl(DS / "suid_aug_labeled.jsonl"))

    # ---- manifest ----
    lines = ["# 样本清单（Sample Manifest）", "",
             "> 分类维度：`来源(source) × 标注(raw/labeled)`。",
             "> 命名按数据源（quasarnix / nl2bash / synthetic / ...），不再是实现细节名。",
             "> `corpus/` 为 git 同步的正式语料；`dataset/` 根目录为工作副本（gitignored）。",
             "", "## 标注", "- **raw**：原始样本，未打标签。", "- **labeled**：Jev 打过标签（软/硬标签）。",
             "", "## 来源", "- **public**：`quasarnix` / `nl2bash` / `benign` / `suspicious` / `gtfobins_payloads`。",
             "- **synthetic**：`synthetic` / `diverse` / `suid_variants` / `longtail`。",
             "", "## 文件清单"]
    for p in sorted(CORPUS.rglob("*.jsonl")):
        n = sum(1 for _ in p.read_text(encoding="utf-8").splitlines() if _.strip())
        lines.append(f"- `{p.relative_to(DS)}`  ({p.stat().st_size/1e6:.2f}MB, {n} 条)")
    (DS / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote dataset/MANIFEST.md")


if __name__ == "__main__":
    main()
