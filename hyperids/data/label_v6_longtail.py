#!/usr/bin/env python3
"""Jev 打标 v6 长尾候选，过滤后输出 labeled JSONL（31 维 soft + hard labels）。"""
from __future__ import annotations
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hyperids import jev_labels, schema

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "dataset" / "corpus" / "raw" / "synthetic" / "v6_longtail_candidates.jsonl"
DST = ROOT / "dataset" / "corpus" / "labeled" / "synthetic" / "v6_longtail_labeled.jsonl"
ACTION_IDX = {aid: i for i, aid in enumerate(schema.action_ids())}


def label_one(row):
    try:
        r = jev_labels.classify(row["text"])
        return {**row, "soft": jev_labels.soft_vector(r), "model": r.model}
    except Exception as e:
        return {**row, "soft": None, "label_error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=str(SRC))
    ap.add_argument("--out", type=str, default=str(DST))
    ap.add_argument("--min_pos_prob", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.src).read_text().splitlines() if l.strip()]
    print(f"candidates: {len(rows)}", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=min(args.workers, 8)) as ex:
        futs = [ex.submit(label_one, r) for r in rows]
        for fut in as_completed(futs):
            results.append(fut.result())

    by_id = {r["text"]: r for r in results}
    kept = 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            r = by_id[row["text"]]
            soft = r.get("soft")
            if not soft or len(soft) != len(schema.all_label_ids()):
                print(f"  skip (no label): {row['text'][:60]}", flush=True)
                continue
            a = row["action"]
            idx = 3 + ACTION_IDX[a]
            if soft[idx] < args.min_pos_prob:
                print(f"  drop pos ({a}={soft[idx]:.2f}): {row['text'][:60]}", flush=True)
                continue
            hard = jev_labels.hard_labels(type('R', (), {'answers': None})() if False else _make_result(soft))
            out_row = {**row, "soft": soft, "verdict": hard[0], "actions": hard[1:], "model": r.get("model")}
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            kept += 1
    print(f"kept {kept}/{len(rows)} -> {out}")


def _make_result(soft):
    from hyperids import jev_client, jev_labels, schema
    # reconstruct a JevResult-like object from soft vector
    class R:
        answers = {}
    r = R()
    vid = schema.verdict_ids()
    probs = {vid[i]: soft[i] for i in range(3)}
    r.answers = {"__verdict__": {"probabilities": probs}}
    for i, aid in enumerate(schema.action_ids()):
        r.answers[aid] = {"noul": soft[3 + i]}
    return r


if __name__ == "__main__":
    main()
