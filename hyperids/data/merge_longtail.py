"""Label + filter + merge long-tail candidates into the v2 KD dataset.

Inputs:
  - public candidates  (dataset/public_candidates.jsonl)   -> label with Jev
  - LLM synth          (dataset/synth_longtail.jsonl)      -> already Jev-labeled
Rules:
  - keep a sample if: (pos/public) any TARGET action soft >= min_pos
                      (neg)        its target action soft <= max_neg
  - new samples go 90% train / 10% val; test is frozen.
Output: dataset/gliclass_v2_aug/{train,val,test,labels_desc}.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from hyperids import schema
from hyperids import jev_labels

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "dataset" / "gliclass_v2_kd"
OUT_DIR = ROOT / "dataset" / "gliclass_v2_aug"
PUBLIC = ROOT / "dataset" / "public_candidates.jsonl"
SYNTH = ROOT / "dataset" / "synth_longtail.jsonl"
LABEL_CACHE = ROOT / "dataset" / "public_labeled.jsonl"

RARE = ["action.cryptomining", "action.brute_force", "action.ransomware",
        "action.registry_persist", "action.timestomp", "action.self_propagate",
        "action.process_inject", "action.account_add"]
AMBIG = ["action.keylog", "action.web_shell", "action.network_scan", "action.bind_shell"]
TARGET_ACTIONS = RARE + AMBIG

IDS = schema.all_label_ids()
VERDICT = IDS[:3]
ACTION_IDX = {a: i for i, a in enumerate(schema.action_ids())}


def sha1(t: str) -> str:
    return hashlib.sha1(t.strip().encode()).hexdigest()


def hard_from_soft(soft):
    verdict = VERDICT[max(range(3), key=lambda i: soft[i])]
    actions = [a for i, a in enumerate(schema.action_ids()) if soft[3 + i] >= 0.5]
    return [verdict] + actions


def label_with_cache(texts, cache_path):
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            try:
                d = json.loads(line)
                done[d["text"]] = d["soft"]
            except Exception:
                pass
    todo = [t for t in texts if t not in done]
    if todo:
        print(f"  Jev labeling {len(todo)} public candidates ...", flush=True)
        for i, t in enumerate(todo):
            try:
                r = jev_labels.classify(t)
                done[t] = jev_labels.soft_vector(r)
            except Exception as e:
                print(f"  label fail {t[:40]!r}: {e}", flush=True)
                continue
            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(todo)}", flush=True)
        with open(cache_path, "w") as f:
            for t in done:
                f.write(json.dumps({"text": t, "soft": done[t]}, ensure_ascii=False) + "\n")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", type=str, default=str(SRC_DIR))
    ap.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--public", type=str, default=str(PUBLIC))
    ap.add_argument("--synth", type=str, default=str(SYNTH))
    ap.add_argument("--min_pos", type=float, default=0.7)
    ap.add_argument("--max_neg", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="cap candidates (smoke)")
    args = ap.parse_args()

    src = Path(args.src_dir)
    train = json.loads((src / "train.json").read_text())
    val = json.loads((src / "val.json").read_text())
    test = json.loads((src / "test.json").read_text())
    existing = {sha1(e["text"]) for e in train + val + test}
    all_labels = list(IDS)

    # gather candidates
    cands = []  # dicts: {text, kind, target_action?}
    for line in Path(args.public).read_text().splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("text"):
            cands.append({"text": d["text"], "kind": "public", "target": d.get("hint")})
    synth_rows = []
    if Path(args.synth).exists():
        for line in Path(args.synth).read_text().splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("text"):
                synth_rows.append(d)
    # dedup + drop existing
    seen = set()
    public_texts = []
    for c in cands:
        k = sha1(c["text"])
        if k in existing or k in seen:
            continue
        seen.add(k)
        public_texts.append(c["text"])

    # label public candidates
    soft_public = label_with_cache(public_texts, LABEL_CACHE)

    # build new samples
    new_samples = []
    # public positives
    for c in cands:
        soft = soft_public.get(c["text"])
        if not soft:
            continue
        if len(soft) != 31:
            continue
        if any(soft[3 + ACTION_IDX[a]] >= args.min_pos for a in TARGET_ACTIONS):
            new_samples.append({"text": c["text"], "true_labels": hard_from_soft(soft),
                                "all_labels": all_labels, "soft_v2": soft})
    # synth (pos + neg)
    for d in synth_rows:
        soft = d.get("soft")
        if not soft or len(soft) != 31:
            continue
        a = d.get("action")
        target = d.get("target")
        if target == "pos":
            keep = soft[3 + ACTION_IDX[a]] >= args.min_pos
        elif target == "neg":
            keep = soft[3 + ACTION_IDX[a]] <= args.max_neg
        else:
            keep = any(soft[3 + ACTION_IDX[x]] >= args.min_pos for x in TARGET_ACTIONS)
        if keep:
            new_samples.append({"text": d["text"], "true_labels": hard_from_soft(soft),
                                "all_labels": all_labels, "soft_v2": soft})

    if args.limit:
        new_samples = new_samples[: args.limit]

    # 90% train / 10% val (shuffle so no action-order bias)
    random.Random(42).shuffle(new_samples)
    n_val = max(1, int(len(new_samples) * 0.1))
    aug_val, aug_train = new_samples[:n_val], new_samples[n_val:]
    out_train = train + aug_train
    out_val = val + aug_val

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", out_train), ("val", out_val), ("test", test)]:
        (out / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False))
    (out / "labels_desc.json").write_text((src / "labels_desc.json").read_text())

    # report support delta
    def support(data):
        c = Counter()
        for e in data:
            for l in e["true_labels"]:
                if l.startswith("action."):
                    c[l] += 1
        return c
    before = support(train)
    after = support(out_train)
    print(f"\nnew samples: {len(new_samples)} (train +{len(aug_train)}, val +{len(aug_val)}) -> {out}")
    for a in TARGET_ACTIONS:
        print(f"  {a:28s} {before.get(a,0):>6} -> {after.get(a,0):>6}")


if __name__ == "__main__":
    main()
