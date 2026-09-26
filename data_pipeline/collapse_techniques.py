"""Collapse technique .NNN sub-techniques to parents (199 -> 132 labels).

Reads dataset/gliclass/{train,val,test}.json + labels_desc.json and writes
dataset/gliclass_collapsed/. The original 199-label dataset is left untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

from schema_v1 import collapsed_label_ids, collapse_labels, label_descriptions

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dataset" / "gliclass"
DST = ROOT / "dataset" / "gliclass_collapsed"


def collapse_list(labels):
    """Collapse child technique -> parent, dedupe, keep first-seen order."""
    seen = set()
    out = []
    for l in labels:
        c = next(iter(collapse_labels([l])))
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def main():
    DST.mkdir(parents=True, exist_ok=True)
    collapsed = collapsed_label_ids()
    desc = label_descriptions()

    for split in ("train", "val", "test"):
        src = json.loads((SRC / f"{split}.json").read_text())
        out = []
        dropped_positive = 0
        for ex in src:
            true = collapse_list(ex["true_labels"])
            all_l = collapse_list(ex["all_labels"])
            dropped_positive += (len(ex["true_labels"]) - len(true))
            out.append({"text": ex["text"], "true_labels": true, "all_labels": all_l})
        (DST / f"{split}.json").write_text(json.dumps(out, ensure_ascii=False))
        print(f"{split}: {len(out)} examples (dropped {dropped_positive} positive child-technique labels)")

    # collapsed labels_desc (order = collapsed_label_ids order)
    labels_desc = [{"label": l, "description": desc[l]} for l in collapsed]
    (DST / "labels_desc.json").write_text(json.dumps(labels_desc, ensure_ascii=False, indent=2))
    print(f"labels_desc: {len(labels_desc)} labels -> {DST/'labels_desc.json'}")


if __name__ == "__main__":
    main()
