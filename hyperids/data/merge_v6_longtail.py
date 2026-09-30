#!/usr/bin/env python3
"""增量合并 v6 长尾样本到 gliclass_v2（test 冻结），输出 gliclass_v3。"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from hyperids import schema

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "dataset" / "gliclass_v2"
OUT = ROOT / "dataset" / "gliclass_v3"
SYNTH = ROOT / "dataset" / "corpus" / "labeled" / "synthetic" / "v6_longtail_labeled.jsonl"
TARGETS = ["action.self_propagate", "action.ransomware", "action.web_shell", "action.keylog"]


def sha1(t: str) -> str:
    return hashlib.sha1(t.strip().encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=str(SRC))
    ap.add_argument("--synth", type=str, default=str(SYNTH))
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    src = Path(args.src)
    train = json.loads((src / "train.json").read_text())
    val = json.loads((src / "val.json").read_text())
    test = json.loads((src / "test.json").read_text())
    existing = {sha1(e["text"]) for e in train + val + test}
    all_labels = list(schema.all_label_ids())

    new_samples = []
    dropped_dup = 0
    for line in Path(args.synth).read_text().splitlines():
        d = json.loads(line)
        text = d.get("text")
        if not text:
            continue
        if sha1(text) in existing:
            dropped_dup += 1
            continue
        new_samples.append({
            "text": text,
            "true_labels": [d["verdict"]] + [a for a in d.get("actions", []) if a in schema.action_ids()],
            "all_labels": all_labels,
            "soft_v2": d.get("soft") if isinstance(d.get("soft"), list) and len(d.get("soft", [])) == len(all_labels) else None,
        })

    random.Random(42).shuffle(new_samples)
    n_val = max(1, int(len(new_samples) * 0.1))
    aug_val, aug_train = new_samples[:n_val], new_samples[n_val:]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(train + aug_train, (out / "train.json").open("w"), ensure_ascii=False)
    json.dump(val + aug_val, (out / "val.json").open("w"), ensure_ascii=False)
    json.dump(test, (out / "test.json").open("w"), ensure_ascii=False)
    (out / "labels_desc.json").write_text((src / "labels_desc.json").read_text())

    def support(data):
        c = Counter()
        for e in data:
            for l in e["true_labels"]:
                if l.startswith("action."):
                    c[l] += 1
        return c
    before = support(train)
    after = support(train + aug_train)
    print(f"new samples: {len(new_samples)} (dup dropped: {dropped_dup})")
    print(f"train {len(train)} -> {len(train)+len(aug_train)}; val {len(val)} -> {len(val)+len(aug_val)}; test frozen {len(test)}")
    for a in TARGETS:
        print(f"  {a:26s} train {before.get(a,0):>5} -> {after.get(a,0):>5}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
