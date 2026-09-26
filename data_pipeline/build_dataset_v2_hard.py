"""Build v2 dataset from collapsed hard labels (no soft file needed).

Runs on the studio using dataset/gliclass_collapsed. Merged actions use
'any source positive' semantics (approximates sum>=0.5; the exact soft-merge
version lives in build_dataset_v2.py and is used locally when soft is present).
"""
from __future__ import annotations

import json
from pathlib import Path

import schema

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SRC = ROOT / "dataset" / "gliclass_collapsed"
DST = ROOT / "dataset" / "gliclass_v2"

OLD_TO_ACTION = {oid: aid for aid, srcs in schema.action_from_old().items() for oid in srcs}


def hard_map(old_hard_labels):
    verdict = None
    actions = set()
    for l in old_hard_labels:
        if l.startswith("risk."):
            verdict = "verdict." + l.split(".", 1)[1]
        elif l in OLD_TO_ACTION:
            actions.add(OLD_TO_ACTION[l])
    if verdict is None:
        verdict = "verdict.suspicious"
    return [verdict] + sorted(actions)


def main():
    DST.mkdir(parents=True, exist_ok=True)
    all_labels = schema.all_label_ids()
    desc = schema.descriptions()
    (DST / "labels_desc.json").write_text(
        json.dumps([{"label": l, "description": desc[l]} for l in all_labels], ensure_ascii=False, indent=2))

    for split in ("train", "val", "test"):
        src = json.loads((SPLIT_SRC / f"{split}.json").read_text())
        out = [{"text": e["text"], "true_labels": hard_map(e["true_labels"]), "all_labels": list(all_labels)} for e in src]
        (DST / f"{split}.json").write_text(json.dumps(out, ensure_ascii=False))
        print(f"{split}: {len(out)}")


if __name__ == "__main__":
    main()
