#!/usr/bin/env python3
"""Batch-score adversarial shell examples against the local HyperIDs v3 model.

This script intentionally reads only adversarial/raw/*.jsonl. It does not read
the project corpus, validation split, test split, or external-eval data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from gliclass import GLiClassModel
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyperids import schema  # noqa: E402

DEFAULT_MODEL_DIR = ROOT / "model" / "checkpoints_gpu" / "final_model_v3"
DEFAULT_RAW_DIR = ROOT / "adversarial" / "raw"
DEFAULT_OUT_DIR = ROOT / "adversarial" / "results"
DEFAULT_THRESHOLDS = ROOT / "configs" / "action_thresholds.json"


def load_candidates(raw_dir: Path) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No JSONL candidates found in {raw_dir}")
    for path in files:
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON at {path}:{lineno}: {exc}") from exc
                cmd = rec.get("command")
                if not isinstance(cmd, str) or not cmd.strip():
                    raise SystemExit(f"Missing non-empty command at {path}:{lineno}")
                norm = " ".join(cmd.split())
                if norm in seen:
                    continue
                seen.add(norm)
                rec = dict(rec)
                rec["_source_file"] = str(path.resolve().relative_to(ROOT))
                rec["_source_line"] = lineno
                rec["_command_sha256"] = hashlib.sha256(cmd.encode("utf-8")).hexdigest()
                records.append(rec)
    return records


def load_thresholds(path: Path, action_ids: list[str]) -> dict[str, float]:
    thresholds = {a: 0.5 for a in action_ids}
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for key, value in raw.items():
                if key in thresholds and isinstance(value, (int, float)):
                    thresholds[key] = float(value)
    return thresholds


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def chunks(xs: list[dict], n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    records = load_candidates(args.raw_dir)
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise SystemExit("No unique candidates to score")

    ids = list(schema.all_label_ids())
    off = schema.group_offsets()
    verdict_ids = ids[off["verdict"][0]:off["verdict"][1]]
    action_ids = ids[off["action"][0]:off["action"][1]]
    thresholds = load_thresholds(args.thresholds, action_ids)

    print(f"Loading model: {args.model_dir}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, min(16, (os.cpu_count() or 4))))
    model = GLiClassModel.from_pretrained(str(args.model_dir)).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), add_prefix_space=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = args.out_dir / "all_annotated.jsonl"
    bypass_path = args.out_dir / "bypasses.jsonl"
    benign_path = args.out_dir / "benign_bypasses.jsonl"
    suspicious_path = args.out_dir / "suspicious_evasions.jsonl"
    detected_path = args.out_dir / "detected_malicious.jsonl"
    tmp_annotated = annotated_path.with_suffix(".jsonl.tmp")
    tmp_bypass = bypass_path.with_suffix(".jsonl.tmp")
    tmp_benign = benign_path.with_suffix(".jsonl.tmp")
    tmp_suspicious = suspicious_path.with_suffix(".jsonl.tmp")
    tmp_detected = detected_path.with_suffix(".jsonl.tmp")

    verdict_counts: Counter[str] = Counter()
    strict_bypass_count = 0
    partial_bypass_count = 0
    detected_count = 0
    category_counts: Counter[str] = Counter()
    strategy_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    bypass_counts_by_category: Counter[str] = Counter()
    bypass_counts_by_strategy: Counter[str] = Counter()
    started = time.time()
    done = 0

    with (
        tmp_annotated.open("w", encoding="utf-8") as ann_f,
        tmp_bypass.open("w", encoding="utf-8") as byp_f,
        tmp_benign.open("w", encoding="utf-8") as benign_f,
        tmp_suspicious.open("w", encoding="utf-8") as susp_f,
        tmp_detected.open("w", encoding="utf-8") as detected_f,
    ):
        for batch in chunks(records, args.batch_size):
            prompts = [
                "".join(f"<<LABEL>>{label}" for label in ids) + "<<SEP>>" + rec["command"]
                for rec in batch
            ]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=args.max_length,
            ).to(device)
            with torch.inference_mode():
                out = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_num_classes=len(ids),
                )
                logits = out.logits.detach().cpu().float()
                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                if logits.shape[0] != len(batch):
                    raise RuntimeError(f"Unexpected batch logits shape: {tuple(logits.shape)}")

            for rec, row in zip(batch, logits):
                raw_scores = {label: float(row[i].item()) for i, label in enumerate(ids)}
                v_logits = torch.tensor([raw_scores[v] for v in verdict_ids], dtype=torch.float32)
                v_probs_t = torch.softmax(v_logits, dim=0)
                verdict_probs = {
                    v: round(float(v_probs_t[i].item()), 6) for i, v in enumerate(verdict_ids)
                }
                verdict = max(verdict_ids, key=lambda v: verdict_probs[v])
                action_probs = {
                    a: round(sigmoid(raw_scores[a]), 6)
                    for a in action_ids
                }
                detected_actions = [
                    a for a in action_ids if action_probs[a] >= thresholds[a]
                ]
                attck = schema.derive_attck(detected_actions)
                bypass = verdict != "verdict.malicious"
                result = dict(rec)
                result["model_dir"] = str(args.model_dir)
                result["model_verdict"] = verdict
                result["model_verdict_probs"] = verdict_probs
                result["model_action_probs"] = action_probs
                result["model_actions"] = detected_actions
                result["model_tactics"] = attck["tactics"]
                result["model_techniques"] = attck["techniques"]
                result["verdict_bypass"] = bypass
                result["bypass_type"] = "verdict_not_malicious" if bypass else "detected_malicious"
                # Keep raw logits for threshold re-analysis without re-running inference.
                result["model_raw_logits"] = {
                    label: round(raw_scores[label], 6) for label in ids
                }
                ann_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                if bypass:
                    byp_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                if verdict == "verdict.benign":
                    strict_bypass_count += 1
                    benign_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                elif verdict == "verdict.suspicious":
                    partial_bypass_count += 1
                    susp_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                else:
                    detected_count += 1
                    detected_f.write(json.dumps(result, ensure_ascii=False) + "\n")

                verdict_counts[verdict] += 1
                cat = str(rec.get("category", "unknown"))
                category_counts[cat] += 1
                strat = str(rec.get("evasion_strategy", "unknown"))
                strategy_counts[strat] += 1
                for a in detected_actions:
                    action_counts[a] += 1
                if bypass:
                    bypass_counts_by_category[cat] += 1
                    bypass_counts_by_strategy[strat] += 1

            done += len(batch)
            elapsed = time.time() - started
            print(
                f"scored {done}/{len(records)}  bypasses={sum(bypass_counts_by_category.values())}  "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            ann_f.flush()
            byp_f.flush()
            benign_f.flush()
            susp_f.flush()
            detected_f.flush()

    tmp_annotated.replace(annotated_path)
    tmp_bypass.replace(bypass_path)
    tmp_benign.replace(benign_path)
    tmp_suspicious.replace(suspicious_path)
    tmp_detected.replace(detected_path)

    total = len(records)
    bypass_total = sum(bypass_counts_by_category.values())
    summary = {
        "model_dir": str(args.model_dir),
        "raw_dir": str(args.raw_dir),
        "unique_candidates": total,
        "bypasses_non_malicious": bypass_total,
        "bypass_rate_non_malicious": round(bypass_total / total, 6) if total else 0.0,
        "strict_benign_bypasses": strict_bypass_count,
        "strict_benign_bypass_rate": round(strict_bypass_count / total, 6) if total else 0.0,
        "partial_suspicious_evasions": partial_bypass_count,
        "detected_malicious": detected_count,
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "bypass_counts_by_category": dict(sorted(bypass_counts_by_category.items())),
        "bypass_counts_by_strategy": dict(sorted(bypass_counts_by_strategy.items())),
        "detected_action_counts": dict(sorted(action_counts.items())),
        "thresholds_path": str(args.thresholds),
        "thresholds_used": thresholds,
        "max_length": args.max_length,
        "elapsed_seconds": round(time.time() - started, 3),
        "annotated_file": str(annotated_path.resolve().relative_to(ROOT)),
        "bypass_file": str(bypass_path.resolve().relative_to(ROOT)),
        "strict_benign_file": str(benign_path.resolve().relative_to(ROOT)),
        "partial_suspicious_file": str(suspicious_path.resolve().relative_to(ROOT)),
        "detected_malicious_file": str(detected_path.resolve().relative_to(ROOT)),
        "definitions": {
            "strict_bypass": "verdict == verdict.benign",
            "partial_evasion": "verdict == verdict.suspicious",
            "non_malicious_bypass": "verdict != verdict.malicious",
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
