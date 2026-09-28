"""Jev-label the deterministic SUID variant candidates (hard labels).

The deterministic generator's `malicious` verdict was found by Jev spot-check to
be 0/30 agreement — Jev correctly says these single-line dual-use commands are
`suspicious` (no SUID context in the text). So we re-label with Jev (325 calls,
cheap) and keep Jev's hard labels (argmax verdict + >=0.5 actions).

Output JSONL: {text, verdict, actions}  (only rows Jev keeps as non-benign)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyperids import jev_labels, jev_client

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "dataset" / "suid_variants.jsonl"
OUT = ROOT / "dataset" / "suid_aug_labeled.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=str(SRC))
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # load candidates, dedup
    test = set(json.loads(l)["text"].strip() for l in open(ROOT / "dataset/external_eval/gtfobins.jsonl"))
    for f in ["payloads", "nl2bash", "cowrie"]:
        test |= set(json.loads(l)["text"].strip() for l in open(ROOT / f"dataset/external_eval/{f}.jsonl"))
    train = set(e["text"].strip() for e in json.load(open(ROOT / "dataset/gliclass_v2/train.json")))

    rows = [json.loads(l) for l in open(args.src)]
    cands = [r for r in rows if r["text"].strip() not in test and r["text"].strip() not in train]
    if args.limit:
        cands = cands[: args.limit]
    print(f"labeling {len(cands)} candidates with Jev ...", flush=True)

    results = jev_client.batch_systemone([r["text"] for r in cands], jev_labels.build_questions(), max_workers=8)

    VERDICTS = list(jev_labels.VERDICT_IDS)
    out_rows = []
    kept = 0
    for r, res in zip(cands, results):
        try:
            vec = jev_labels.soft_vector(res)
        except Exception:
            continue
        if len(vec) != 31:
            continue
        verdict = VERDICTS[max(range(3), key=lambda i: vec[i])]
        actions = [a for i, a in enumerate(jev_labels.ACTION_IDS) if vec[3 + i] >= 0.5]
        if verdict == "verdict.benign":
            continue  # drop Jev-benign (would be a wrong label for the aug set)
        out_rows.append({"text": r["text"], "verdict": verdict, "actions": actions})
        kept += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"kept {kept}/{len(cands)} (non-benign) -> {args.out}")


if __name__ == "__main__":
    main()
