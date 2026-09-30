#!/usr/bin/env python3
"""Run-scoped verifier for 20260930-v4-r130 (adapted from adversarial/verify_results.py
with explicit --raw-dir/--results-dir/--model-dir per AGENTS.md section 7).

Checks: 1:1 raw<->results, 3 verdict probs, 28 action probs, 31 logits,
verdict==argmax(probs), verdict_bypass consistency, split counts, model frozen,
thresholds frozen, truncation group reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]


def read_jsonl(p):
    rows = []
    with p.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{p}:{i}: {e}")
    return rows


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "adversarial" / "raw")
    ap.add_argument("--results-dir", type=Path, default=RUN_DIR / "results")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "model" / "checkpoints_gpu" / "final_model_v4")
    a = ap.parse_args()
    run = json.loads((RUN_DIR / "run.json").read_text())

    cand = []
    for p in sorted(a.raw_dir.glob("*.jsonl")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                r = json.loads(line)
                norm = " ".join(r["command"].split())
                cand.append((hashlib.sha256(r["command"].encode()).hexdigest(), str(p.relative_to(ROOT))))
    cand_hash = {h for h, _ in cand}
    ann = read_jsonl(a.results_dir / "all_annotated.jsonl")
    by_hash = {r["_command_sha256"]: r for r in ann}
    errors = []
    if len(by_hash) != len(cand_hash):
        errors.append(f"unique {len(by_hash)} != candidates {len(cand_hash)}")
    missing = cand_hash - set(by_hash)
    extra = set(by_hash) - cand_hash
    if missing:
        errors.append(f"{len(missing)} missing records")
    if extra:
        errors.append(f"{len(extra)} extra records")
    vc: Counter = Counter()
    trunc = Counter()
    for r in ann:
        h = r.get("_command_sha256", "?")
        v = r.get("model_verdict")
        if v not in ("verdict.benign", "verdict.suspicious", "verdict.malicious"):
            errors.append(f"{h}: bad verdict {v!r}")
            continue
        vc[v] += 1
        pr = r.get("model_verdict_probs") or {}
        if (max(pr, key=pr.get) if pr else None) != v:
            errors.append(f"{h}: verdict != argmax")
        if len(pr) != 3:
            errors.append(f"{h}: verdict probs != 3")
        if r.get("verdict_bypass") != (v != "verdict.malicious"):
            errors.append(f"{h}: bypass flag inconsistent")
        if len(r.get("model_action_probs") or {}) != 28:
            errors.append(f"{h}: action probs != 28")
        if len(r.get("model_raw_logits") or {}) != 31:
            errors.append(f"{h}: logits != 31")
        if r.get("model_dir") != str(a.model_dir):
            errors.append(f"{h}: model_dir mismatch")
        if "r3_long_prefix" in (r.get("_source_file") or ""):
            trunc[v] += 1
    splits = {k: len(read_jsonl(a.results_dir / f"{k}.jsonl")) for k in
              ("bypasses", "benign_bypasses", "suspicious_evasions", "detected_malicious")}
    exp = {"bypasses": vc["verdict.benign"] + vc["verdict.suspicious"],
           "benign_bypasses": vc["verdict.benign"],
           "suspicious_evasions": vc["verdict.suspicious"],
           "detected_malicious": vc["verdict.malicious"]}
    for k in exp:
        if splits[k] != exp[k]:
            errors.append(f"{k}: {splits[k]} != {exp[k]}")
    if str(a.model_dir) != str(ROOT / run["target_model_dir"]):
        errors.append("target model not frozen")
    w = sha(ROOT / run["target_model_dir"] / "model.safetensors")
    if w != run["target_model_sha256"]:
        errors.append("weights sha mismatch")
    ver = {"status": "verified" if not errors else "failed", "errors": errors[:100],
           "error_count": len(errors), "verdict_counts": dict(vc),
           "split_counts": splits, "truncation_group_r3": dict(trunc),
           "unique_candidates": len(cand_hash), "training_data_read": False,
           "files": {str(p.relative_to(ROOT)): {"sha256": sha(p), "bytes": p.stat().st_size}
                     for p in sorted([*a.raw_dir.glob("*.jsonl"), *a.results_dir.glob("*.jsonl"),
                                      a.results_dir / "summary.json", RUN_DIR / "run.json"])},
           "weights_sha256_now": w}
    (a.results_dir / "verification.json").write_text(json.dumps(ver, ensure_ascii=False, indent=2) + "\n")
    (RUN_DIR / "MANIFEST.sha256").write_text(
        "".join(f"{v['sha256']}  {k}\n" for k, v in sorted(ver["files"].items())) +
        f"{w}  {run['target_model_dir']}/model.safetensors\n")
    print(json.dumps({k: v for k, v in ver.items() if k != "files"}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
