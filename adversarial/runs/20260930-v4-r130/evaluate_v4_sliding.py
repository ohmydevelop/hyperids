#!/usr/bin/env python3
"""v1.3.0 run 20260930-v4-r130: production-faithful sliding-window eval of
final_model_v4 on the FROZEN v3 raw set (read-only). No training data read.

Inference mirrors hyperids/predict.py: tokenize full command, 120-token
segments stepped by 90 (+ tail segment), per-segment logits, merge by max
per label (verdict = most-malicious segment, action = max). ids capped
at 512 exactly like production. Thresholds: all 0.5 (file absent).
Result contract matches evaluate_adversarial.py (3 verdict probs,
28 action probs, 31 raw logits) + n_segments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch

RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(ROOT))
from hyperids import schema  # noqa: E402
from gliclass import GLiClassModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

FROZEN_RAW = ROOT / "adversarial" / "raw"
WINDOW, STRIDE, CAP = 120, 90, 512


def load_candidates(raw_dir: Path):
    records, seen = [], set()
    for path in sorted(raw_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cmd = rec.get("command")
                if not isinstance(cmd, str) or not cmd.strip():
                    raise SystemExit(f"Missing command at {path}:{lineno}")
                norm = " ".join(cmd.split())
                if norm in seen:
                    continue
                seen.add(norm)
                rec = dict(rec)
                rec["_source_file"] = str(path.resolve().relative_to(ROOT))
                rec["_source_line"] = lineno
                rec["_command_sha256"] = hashlib.sha256(cmd.encode()).hexdigest()
                records.append(rec)
    return records


def segments(cmd_ids):
    if len(cmd_ids) <= WINDOW:
        return [cmd_ids]
    segs, start = [], 0
    while True:
        segs.append(cmd_ids[start:start + WINDOW])
        if start + WINDOW >= len(cmd_ids):
            break
        start += STRIDE
    if segs[-1][-1] != cmd_ids[-1]:
        segs.append(cmd_ids[-WINDOW:])
    return segs


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x < 0 else math.exp(x) / (1.0 + math.exp(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    run = json.loads((RUN_DIR / "run.json").read_text())
    model_dir = ROOT / run["target_model_dir"]
    out_dir = RUN_DIR / "results"

    records = load_candidates(FROZEN_RAW)
    print(f"candidates: {len(records)}", flush=True)
    ids = list(schema.all_label_ids())
    off = schema.group_offsets()
    verdict_ids = ids[off["verdict"][0]:off["verdict"][1]]
    action_ids = ids[off["action"][0]:off["action"][1]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, min(16, (os.cpu_count() or 4))))
    model = GLiClassModel.from_pretrained(str(model_dir)).to(device).eval()
    tok = AutoTokenizer.from_pretrained(str(model_dir), add_prefix_space=True)
    prefix = "".join(f"<<LABEL>>{l}" for l in ids) + "<<SEP>>"
    pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    sep_id = tok.sep_token_id

    # Build segment work list: (rec_idx, input_ids)
    segs_of, work = [], []
    for i, rec in enumerate(records):
        cmd_ids = tok(rec["command"], add_special_tokens=False)["input_ids"]
        sg = segments(cmd_ids)
        segs_of.append(len(sg))
        for s in sg:
            work.append((i, (pre_ids + s + [sep_id])[:CAP]))
    print(f"segments total: {len(work)}", flush=True)

    merged = [[-1e9] * len(ids) for _ in records]
    t0 = time.time()
    for b in range(0, len(work), args.batch_size):
        chunk = work[b:b + args.batch_size]
        L = max(len(x[1]) for x in chunk)
        inp = torch.tensor([x[1] + [tok.pad_token_id] * (L - len(x[1])) for x in chunk],
                           dtype=torch.long, device=device)
        msk = (inp != tok.pad_token_id).long()
        with torch.inference_mode():
            lg = model(input_ids=inp, attention_mask=msk, max_num_classes=len(ids)).logits.detach().cpu().float()
        if lg.ndim == 1:
            lg = lg.unsqueeze(0)
        for (i, _), row in zip(chunk, lg):
            r = row.tolist()
            m = merged[i]
            for j, v in enumerate(r):
                if v > m[j]:
                    m[j] = v
        if (b // args.batch_size + 1) % 20 == 0:
            print(f"  batch {b//args.batch_size+1}/{(len(work)+args.batch_size-1)//args.batch_size} "
                  f"{time.time()-t0:.0f}s", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: out_dir / f"{k}.jsonl" for k in
             ("all_annotated", "bypasses", "benign_bypasses", "suspicious_evasions", "detected_malicious")}
    fps = {k: p.open("w", encoding="utf-8") for k, p in paths.items()}
    vc: Counter = Counter()
    by_src: dict[str, Counter] = {}
    n_strict = n_part = n_det = 0
    for rec, m, ns in zip(records, merged, segs_of):
        raw = {l: float(m[j]) for j, l in enumerate(ids)}
        vl = torch.tensor([raw[v] for v in verdict_ids])
        vp = torch.softmax(vl, dim=0)
        vprobs = {v: round(float(vp[i].item()), 6) for i, v in enumerate(verdict_ids)}
        verdict = max(verdict_ids, key=lambda v: vprobs[v])
        aprobs = {a: round(sigmoid(raw[a]), 6) for a in action_ids}
        acts = [a for a in action_ids if aprobs[a] >= 0.5]
        att = schema.derive_attck(acts)
        byp = verdict != "verdict.malicious"
        res = dict(rec)
        res.update({"model_dir": str(model_dir), "model_verdict": verdict,
                    "model_verdict_probs": vprobs, "model_action_probs": aprobs,
                    "model_actions": acts, "model_tactics": att["tactics"],
                    "model_techniques": att["techniques"], "verdict_bypass": byp,
                    "bypass_type": "verdict_not_malicious" if byp else "detected_malicious",
                    "model_raw_logits": {l: round(raw[l], 6) for l in ids},
                    "n_segments": ns})
        fps["all_annotated"].write(json.dumps(res, ensure_ascii=False) + "\n")
        vc[verdict] += 1
        by_src.setdefault(res["_source_file"], Counter())[verdict] += 1
        if byp:
            fps["bypasses"].write(json.dumps(res, ensure_ascii=False) + "\n")
        if verdict == "verdict.benign":
            n_strict += 1
            fps["benign_bypasses"].write(json.dumps(res, ensure_ascii=False) + "\n")
        elif verdict == "verdict.suspicious":
            n_part += 1
            fps["suspicious_evasions"].write(json.dumps(res, ensure_ascii=False) + "\n")
        else:
            n_det += 1
            fps["detected_malicious"].write(json.dumps(res, ensure_ascii=False) + "\n")
    for f in fps.values():
        f.close()
    n = len(records)
    summary = {
        "run_id": run["run_id"], "model_dir": str(model_dir),
        "raw_dir": str(FROZEN_RAW), "unique_candidates": n,
        "bypasses_non_malicious": vc["verdict.benign"] + vc["verdict.suspicious"],
        "bypass_rate_non_malicious": round((vc["verdict.benign"] + vc["verdict.suspicious"]) / n, 6),
        "strict_benign_bypasses": n_strict,
        "strict_benign_bypass_rate": round(n_strict / n, 6),
        "partial_suspicious_evasions": n_part, "detected_malicious": n_det,
        "verdict_counts": dict(vc),
        "by_source_file": {k: dict(v) for k, v in sorted(by_src.items())},
        "inference": run["inference"], "eval_max_length": run["eval_max_length"],
        "thresholds": "all-0.5 (file absent)",
        "elapsed_seconds": round(time.time() - t0, 1),
        "definitions": "strict=model_verdict==verdict.benign; partial==verdict.suspicious; wide=verdict!=verdict.malicious",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
