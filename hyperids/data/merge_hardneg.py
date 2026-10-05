#!/usr/bin/env python3
"""基于最强模型仲裁结果，回灌 hard-negative 样本（gt=malicious & arb=malicious）。

- 只回灌“双标签一致恶意”的样本，verdict 强制 malicious，actions 用 Jev actions。
- 去重（对 src train/val/test 的 sha1 text）+ 来源记录。
- test 冻结，输出新数据集目录。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from hyperids import schema

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "dataset" / "gliclass_v3"
ARB = ROOT / "adversarial" / "runs" / "20260930-v5-hardneg-audit" / "arbitration_gpt_5_6_sol.jsonl"
LABELED = ROOT / "adversarial" / "labeled.jsonl"
OUT = ROOT / "dataset" / "gliclass_v4"


def sha1(t: str) -> str:
    return hashlib.sha1(t.strip().encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=str(SRC))
    ap.add_argument("--arb", type=str, default=str(ARB))
    ap.add_argument("--labeled", type=str, default=str(LABELED))
    ap.add_argument("--prepared", type=str, default="", help="prepared JSONL {text, verdict, actions} (skip arb+labeled)")
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--val_ratio", type=float, default=0.1)
    args = ap.parse_args()

    src = Path(args.src)
    train = json.loads((src / "train.json").read_text())
    val = json.loads((src / "val.json").read_text())
    test = json.loads((src / "test.json").read_text())
    existing = {sha1(e["text"]) for e in train + val + test}
    all_labels = list(schema.all_label_ids())
    action_ids = set(schema.action_ids())

    new_samples = []
    dropped_dup = 0
    skipped_nojev = 0
    if args.prepared:
        for line in Path(args.prepared).read_text().splitlines():
            d = json.loads(line)
            text = d.get("text")
            if not text or sha1(text) in existing:
                dropped_dup += 1
                continue
            actions = [a for a in d.get("actions", []) if a in action_ids]
            new_samples.append({"text": text, "true_labels": [d.get("verdict", "verdict.malicious")] + actions,
                                "all_labels": all_labels, "source": "adversarial_hardneg_gt_malicious_gpt_malicious",
                                "jev_verdict": None})
        random.Random(42).shuffle(new_samples)
        n_val = max(1, int(len(new_samples) * args.val_ratio))
        aug_val, aug_train = new_samples[:n_val], new_samples[n_val:]
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        json.dump(train + aug_train, (out / "train.json").open("w"), ensure_ascii=False)
        json.dump(val + aug_val, (out / "val.json").open("w"), ensure_ascii=False)
        json.dump(test, (out / "test.json").open("w"), ensure_ascii=False)
        (out / "labels_desc.json").write_text((src / "labels_desc.json").read_text())
        print(f"prepared hardneg: {len(new_samples)} (dup={dropped_dup}); train {len(train)} -> {len(train)+len(aug_train)}; test frozen {len(test)}")
        return
    jev_by_text = {}
    for line in Path(args.labeled).read_text().splitlines():
        d = json.loads(line)
        jev_by_text[d["text"]] = d
    for line in Path(args.arb).read_text().splitlines():
        r = json.loads(line)
        if r.get("gt") != "malicious" or r.get("arb_verdict") != "malicious":
            continue
        text = r["text"]
        if sha1(text) in existing:
            dropped_dup += 1
            continue
        j = jev_by_text.get(text)
        if not j:
            skipped_nojev += 1
            continue
        actions = [a for a in j.get("actions", []) if a in action_ids]
        new_samples.append({
            "text": text,
            "true_labels": ["verdict.malicious"] + actions,
            "all_labels": all_labels,
            "source": "adversarial_hardneg_gt_malicious_gpt_malicious",
            "jev_verdict": j.get("verdict"),
        })

    random.Random(42).shuffle(new_samples)
    n_val = max(1, int(len(new_samples) * args.val_ratio))
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
    print(f"hardneg selected: {len(new_samples)} (dup={dropped_dup}, nojev={skipped_nojev})")
    print(f"train {len(train)} -> {len(train)+len(aug_train)}; val {len(val)} -> {len(val)+len(aug_val)}; test frozen {len(test)}")
    print("action support delta (train):")
    changed = set(before) | set(after)
    for a in sorted(changed, key=lambda x: str(x)):
        if before.get(a, 0) != after.get(a, 0):
            print(f"  {a:26s} {before.get(a,0):>6} -> {after.get(a,0):>6}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
