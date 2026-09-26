"""Build the v2 (verdict 3 + action 28 = 31 labels) GLiClass dataset.

Derives v2 labels deterministically from the existing Jev 199-dim soft labels
(soft_labels_50k.jsonl) plus the same train/val/test text split as
dataset/gliclass_collapsed. MITRE tactic/technique are NOT labels; they are
derived from actions via schema.action_attck() at inference.
"""
from __future__ import annotations

import json
from pathlib import Path

from schema_v1 import all_label_ids as OLD_IDS
import schema

ROOT = Path(__file__).resolve().parents[1]
SOFT_JSONL = ROOT / "dataset" / "soft_labels_50k.jsonl"
SPLIT_SRC = ROOT / "dataset" / "gliclass_collapsed"
DST = ROOT / "dataset" / "gliclass_v2"

OLD = OLD_IDS()
OLD_IDX = {oid: i for i, oid in enumerate(OLD)}


def _source_indices(oid: str):
    """indices of old soft vector for a source id (incl. technique sub-techniques)."""
    return [i for i, x in enumerate(OLD) if x == oid or x.startswith(oid + ".")]


# precompute source index sets
VERDICT_SRC = {vid: _source_indices(oid) for vid, oid in schema.verdict_from_old().items()}
ACTION_SRC = {aid: [j for oid in sources for j in _source_indices(oid)]
              for aid, sources in schema.action_from_old().items()}


def soft_to_v2(old_soft) -> list[float]:
    v = [sum(old_soft[j] for j in VERDICT_SRC[vid]) for vid in schema.verdict_ids()]
    a = [min(1.0, sum(old_soft[j] for j in ACTION_SRC[aid])) for aid in schema.action_ids()]
    return v + a


def hard_from_v2(v2soft) -> list[str]:
    v = v2soft[:3]
    verdict = [schema.verdict_ids()[max(range(3), key=lambda i: v[i])]]
    actions = [aid for i, aid in enumerate(schema.action_ids()) if v2soft[3 + i] >= 0.5]
    return verdict + actions


def hard_map(old_hard_labels) -> list[str]:
    """Fallback for the ~4% texts missing from the soft file."""
    old_to_action = {oid: aid for aid, srcs in schema.action_from_old().items() for oid in srcs}
    verdict = None
    actions = set()
    for l in old_hard_labels:
        if l.startswith("risk."):
            verdict = "verdict." + l.split(".", 1)[1]
        elif l in old_to_action:
            actions.add(old_to_action[l])
    if verdict is None:
        verdict = "verdict.suspicious"
    return [verdict] + sorted(actions)


def main():
    DST.mkdir(parents=True, exist_ok=True)
    soft_by_text = {}
    with open(SOFT_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            soft_by_text[d["text"]] = d["soft"]

    all_labels = schema.all_label_ids()
    desc = schema.descriptions()
    (DST / "labels_desc.json").write_text(
        json.dumps([{"label": l, "description": desc[l]} for l in all_labels], ensure_ascii=False, indent=2))

    for split in ("train", "val", "test"):
        src = json.loads((SPLIT_SRC / f"{split}.json").read_text())
        out, with_soft, missing = [], 0, 0
        for ex in src:
            soft = soft_by_text.get(ex["text"])
            if soft is not None:
                v2 = soft_to_v2(soft)
                true = hard_from_v2(v2)
                with_soft += 1
            else:
                true = hard_map(ex["true_labels"])
                v2 = None
                missing += 1
            item = {"text": ex["text"], "true_labels": true, "all_labels": list(all_labels)}
            if v2 is not None:
                item["soft_v2"] = v2
            out.append(item)
        (DST / f"{split}.json").write_text(json.dumps(out, ensure_ascii=False))
        print(f"{split}: {len(out)} examples (soft {with_soft}, hard-fallback {missing})")

    print(f"labels_desc: {len(all_labels)} labels -> {DST/'labels_desc.json'}")


if __name__ == "__main__":
    main()
