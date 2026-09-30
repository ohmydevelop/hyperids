#!/usr/bin/env python3
"""Verify that every adversarial candidate has a complete v3 detection record."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluate_adversarial import DEFAULT_MODEL_DIR, load_candidates

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "adversarial" / "raw"
OUT_DIR = ROOT / "adversarial" / "results"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: {exc}") from exc
    return rows


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    candidates = load_candidates(RAW_DIR)
    candidate_by_hash = {r["_command_sha256"]: r for r in candidates}
    annotated = read_jsonl(OUT_DIR / "all_annotated.jsonl")
    by_hash = {r["_command_sha256"]: r for r in annotated}

    errors: list[str] = []
    if len(by_hash) != len(candidates):
        errors.append(f"annotated unique hashes {len(by_hash)} != candidates {len(candidates)}")
    missing = sorted(set(candidate_by_hash) - set(by_hash))
    extra = sorted(set(by_hash) - set(candidate_by_hash))
    if missing:
        errors.append(f"{len(missing)} candidates have no detection record")
    if extra:
        errors.append(f"{len(extra)} detection records have no source candidate")

    verdicts = {"verdict.benign": 0, "verdict.suspicious": 0, "verdict.malicious": 0}
    for row in annotated:
        h = row.get("_command_sha256", "<missing>")
        verdict = row.get("model_verdict")
        if verdict not in verdicts:
            errors.append(f"{h}: invalid model_verdict={verdict!r}")
            continue
        verdicts[verdict] += 1
        probs = row.get("model_verdict_probs") or {}
        expected = max(probs, key=probs.get) if probs else None
        if expected != verdict:
            errors.append(f"{h}: verdict {verdict} != argmax {expected}")
        if row.get("verdict_bypass") != (verdict != "verdict.malicious"):
            errors.append(f"{h}: inconsistent verdict_bypass")
        for key in ("model_action_probs", "model_actions", "model_raw_logits"):
            if key not in row:
                errors.append(f"{h}: missing {key}")
        if len(row.get("model_raw_logits") or {}) != 31:
            errors.append(f"{h}: raw logits do not contain 31 labels")
        if len(row.get("model_action_probs") or {}) != 28:
            errors.append(f"{h}: action probs do not contain 28 labels")

    split_counts = {
        "bypasses": len(read_jsonl(OUT_DIR / "bypasses.jsonl")),
        "benign_bypasses": len(read_jsonl(OUT_DIR / "benign_bypasses.jsonl")),
        "suspicious_evasions": len(read_jsonl(OUT_DIR / "suspicious_evasions.jsonl")),
        "detected_malicious": len(read_jsonl(OUT_DIR / "detected_malicious.jsonl")),
    }
    expected_splits = {
        "bypasses": verdicts["verdict.benign"] + verdicts["verdict.suspicious"],
        "benign_bypasses": verdicts["verdict.benign"],
        "suspicious_evasions": verdicts["verdict.suspicious"],
        "detected_malicious": verdicts["verdict.malicious"],
    }
    for name, got in split_counts.items():
        if got != expected_splits[name]:
            errors.append(f"{name}: count {got} != expected {expected_splits[name]}")

    requirements = {
        "unique_adversarial_samples": len(candidates),
        "candidate_count_gte_1000": len(candidates) >= 1000,
        "non_malicious_bypasses": expected_splits["bypasses"],
        "non_malicious_bypasses_gte_1000": expected_splits["bypasses"] >= 1000,
        "strict_benign_bypasses": expected_splits["benign_bypasses"],
        "strict_benign_bypasses_gte_1000": expected_splits["benign_bypasses"] >= 1000,
        "all_candidates_have_detection": not missing and not extra and len(by_hash) == len(candidates),
        "model_matches_current_v3": str(DEFAULT_MODEL_DIR) == str(next(iter(annotated)).get("model_dir")) if annotated else False,
    }
    verification = {
        "status": "verified" if not errors else "failed",
        "errors": errors[:100],
        "error_count": len(errors),
        "requirements": requirements,
        "verdict_counts": verdicts,
        "split_counts": split_counts,
        "files": {
            str(p.relative_to(ROOT)): {
                "sha256": file_sha256(p),
                "bytes": p.stat().st_size,
            }
            for p in sorted([*RAW_DIR.glob("*.jsonl"), *OUT_DIR.glob("*.jsonl"), OUT_DIR / "summary.json"])
        },
    }
    out = OUT_DIR / "verification.json"
    out.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
