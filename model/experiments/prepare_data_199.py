"""Convert the Jev-labeled pool into GLiClass training data.

GLiClass expects a JSON list of {"text": ..., "all_labels": [...]}, optionally
plus a labels-description JSON list of {"label": ..., "description": ...}.
Split is stratified by risk so train/val/test keep the same risk distribution.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from schema_v1 import all_label_ids, label_descriptions, group_offsets

ROOT = Path(__file__).resolve().parents[1]
LABEL_JSONL = ROOT / "dataset" / "soft_labels_50k.jsonl"
OUT_DIR = ROOT / "dataset" / "gliclass"
RISK_IDS = all_label_ids()[: group_offsets()["risk"][1]]


def load_valid_rows(path: Path) -> list[dict]:
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        labels = [l for l in r.get("labels") or [] if l in set(all_label_ids())]
        if not labels:
            continue  # skip failed/empty
        rows.append({"text": r["text"], "all_labels": labels, "risk": next((l for l in labels if l.startswith("risk.")), None)})
    return rows


def stratified_split(rows: list[dict], val_ratio=0.08, test_ratio=0.08, seed=42):
    import random
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["risk"]].append(r)
    train, val, test = [], [], []
    rng = random.Random(seed)
    for risk, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        test += items[:n_test]
        val += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]
    rng.shuffle(train)
    return train, val, test


def _make_glicexample(r: dict, rng, all_ids: list[str], neg_k: int = 25) -> dict:
    """true_labels = positives; all_labels = positives + sampled negatives.

    The two non-chosen risk labels are always added as explicit negatives;
    remaining negatives are sampled from the rest of the label space.
    """
    true = list(r["all_labels"])
    risk_set = {"risk.benign", "risk.suspicious", "risk.malicious"}
    negatives = [lid for lid in risk_set if lid not in true]
    pool = [lid for lid in all_ids if lid not in true and lid not in negatives]
    rng.shuffle(pool)
    negatives += pool[:neg_k]
    return {"text": r["text"], "true_labels": true, "all_labels": true + negatives}


def build(label_jsonl: Path = LABEL_JSONL, out_dir: Path = OUT_DIR, neg_k: int = 25):
    import random
    rows = load_valid_rows(label_jsonl)
    print(f"valid labeled rows: {len(rows)}")
    train, val, test = stratified_split(rows)

    descs = [{"label": lid, "description": label_descriptions()[lid]} for lid in all_label_ids()]

    rng = random.Random(42)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val), ("test", test)):
        data = [_make_glicexample(r, rng, list(all_label_ids()), neg_k) for r in split]
        with open(out_dir / f"{name}.json", "w") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"  {name}: {len(data)}")
    with open(out_dir / "labels_desc.json", "w") as f:
        json.dump(descs, f, ensure_ascii=False)
    print(f"→ {out_dir}/ (train/val/test.json + labels_desc.json)")

    # risk distribution sanity
    for name, split in (("train", train), ("val", val), ("test", test)):
        c = Counter(r["risk"] for r in split)
        print(f"  risk[{name}]:", dict(c))


if __name__ == "__main__":
    build()
