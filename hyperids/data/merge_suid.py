"""Merge Jev-labeled SUID variants into the v2 train split (hard labels).

Frozen: val/test unchanged (external test set stays untouched too).
New: dataset/gliclass_v2_suid/{train,val,test,labels_desc}.json
"""
from __future__ import annotations

import json
from pathlib import Path

from hyperids import schema

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "dataset" / "gliclass_v2"
LABELED = ROOT / "dataset" / "suid_aug_labeled.jsonl"
OUT = ROOT / "dataset" / "gliclass_v2_suid"

ALL_LABELS = list(schema.all_label_ids())


def main():
    train = json.loads((SRC / "train.json").read_text())
    val = json.loads((SRC / "val.json").read_text())
    test = json.loads((SRC / "test.json").read_text())
    existing = {e["text"].strip() for e in train + val + test}

    added = 0
    for line in LABELED.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        text = r["text"].strip()
        if text in existing:
            continue
        labels = [r["verdict"]] + [a for a in r["actions"] if a in ALL_LABELS]
        train.append({"text": text, "true_labels": labels, "all_labels": list(ALL_LABELS)})
        existing.add(text)
        added += 1

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.json").write_text(json.dumps(train, ensure_ascii=False))
    (OUT / "val.json").write_text(json.dumps(val, ensure_ascii=False))
    (OUT / "test.json").write_text(json.dumps(test, ensure_ascii=False))
    (OUT / "labels_desc.json").write_text((SRC / "labels_desc.json").read_text())
    print(f"added {added} samples -> train now {len(train)} (val {len(val)}, test {len(test)}) -> {OUT}")


if __name__ == "__main__":
    main()
