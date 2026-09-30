#!/usr/bin/env python3
"""Audit v5 verdicts on frozen red-team labeled set; emit hard-negative candidates.

Only reads adversarial/labeled.jsonl (frozen gt by Jev). Does NOT read train/val/test.
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "adversarial" / "runs" / "20260930-v5-hardneg-audit"
LABELED = ROOT / "adversarial" / "labeled.jsonl"
MODEL_DIR = ROOT / "model" / "checkpoints_gpu" / "final_model_v5"

def main():
    from hyperids.predict import Predictor
    pred = Predictor(model_dir=MODEL_DIR, thresholds_path=ROOT/"configs"/"action_thresholds.json")
    rows = [json.loads(l) for l in LABELED.read_text().splitlines() if l.strip()]

    gt_dist = Counter(r.get("gt") for r in rows)
    mv_by_gt = defaultdict(Counter)
    cands = []
    for i, r in enumerate(rows):
        text = r["text"]
        try:
            p = pred.predict(text)
        except Exception as e:
            p = {"verdict": "error", "verdict_probs": {}, "actions": [], "action_probs": {}, "tactics": [], "techniques": [], "_error": repr(e)}
        gt = r.get("gt")
        mv = p["verdict"].replace("verdict.", "") if isinstance(p.get("verdict"), str) else str(p.get("verdict"))
        mv_by_gt[gt][mv] += 1
        if gt == "malicious" and mv != "malicious":
            cands.append({
                "text": text,
                "gt": gt,
                "model_verdict": p["verdict"],
                "model_verdict_probs": p.get("verdict_probs", {}),
                "model_actions": p.get("actions", []),
                "model_action_probs": p.get("action_probs", {}),
                "source_file": "adversarial/labeled.jsonl",
                "source_line": i + 1,
            })
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    RUN.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": RUN.name,
        "model_dir": str(MODEL_DIR),
        "total": len(rows),
        "gt_distribution": dict(gt_dist),
        "model_verdict_by_gt": {k: dict(v) for k, v in sorted(mv_by_gt.items(), key=lambda kv: str(kv[0]))},
        "hard_neg_candidates": len(cands),
        "hard_neg_suspicious": sum(1 for c in cands if c["model_verdict"] == "verdict.suspicious"),
        "hard_neg_benign": sum(1 for c in cands if c["model_verdict"] == "verdict.benign"),
        "note": "gt=malicious but v5 verdict != malicious are hard-negative training candidates; do NOT auto-merge without user authorization",
    }
    (RUN / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (RUN / "hard_neg_candidates.jsonl").open("w") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
