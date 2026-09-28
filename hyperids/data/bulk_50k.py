"""Scale the labeled dataset to ~50K — resumable, checkpointed.

Phase 1  generate : LLM produces candidates (coverage synthesis + bulk diverse)
                    -> dataset/candidates_50k.jsonl  (dedupe by text hash)
Phase 2  label    : Jev soft-labels every candidate, chunked & resumable
                    -> dataset/soft_labels_50k.jsonl
Phase 3  finalize : dedupe + train/val/test split -> dataset/soft_labels.parquet

Resume: re-running `label` skips candidates already present in the label file.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from hyperids import jev_client, jev_labels_legacy, llm, prompts
from hyperids.seeds import ALL_SEEDS

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "dataset"
CAND_PATH = DS / "candidates_50k.jsonl"
LABEL_PATH = DS / "soft_labels_50k.jsonl"
OUT_PATH = DS / "soft_labels.parquet"

GEN_MODEL = "feature/flash"       # fastest reliable model for bulk generation
GEN_WORKERS = 12
LABEL_WORKERS = 20


def _sha(text: str) -> str:
    return hashlib.sha1(text.strip().encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Phase 1 — generation
# --------------------------------------------------------------------------- #
def _load_combos() -> list[list[str]]:
    combos: set[tuple] = set()
    for p in [DS / "seed_hard_jev.parquet"]:
        if p.exists():
            df = pd.read_parquet(p)
            for _, r in df.iterrows():
                if len(r["labels"]) > 0:
                    combos.add(tuple(r["labels"]))
    if (DS / "soft_labels.parquet").exists():
        df = pd.read_parquet(DS / "soft_labels.parquet")
        for _, r in df.iterrows():
            if len(r["labels"]) > 0:
                combos.add(tuple(r["labels"]))
    return [list(c) for c in combos]


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _synth_one(labels: list[str], n: int) -> list[dict]:
    parsed, _ = llm.chat_json(prompts.synthesize_prompt(labels, n=n), model=GEN_MODEL,
                              temperature=1.0, max_tokens=4096)
    out = []
    for s in (parsed or {}).get("samples") or []:
        if str(s).strip():
            out.append({"text": str(s).strip(), "source": "synthetic", "hint": list(labels)})
    return out


def _diverse_one(n: int) -> list[dict]:
    parsed, _ = llm.chat_json(prompts.diverse_prompt(n), model=GEN_MODEL,
                              temperature=1.0, max_tokens=8192)
    out = []
    for s in (parsed or {}).get("commands") or []:
        if str(s).strip():
            out.append({"text": str(s).strip(), "source": "diverse", "hint": []})
    return out


def _run_pool(jobs, worker_fn, workers, desc, every=50):
    results: list[list[dict]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(worker_fn, *j): j for j in jobs}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                print("  ! job failed:", str(e)[:80])
            done += 1
            if done % every == 0:
                print(f"  {desc}: {done}/{len(jobs)}")
    flat = [x for sub in results for x in sub]
    return flat


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.add(json.loads(line)["id"])
                except Exception:
                    pass
    return out


def generate(target: int = 50000, coverage_per: int = 10, batch_size: int = 30,
             workers: int = 8):
    """Incremental + resumable generation. Appends to CAND_PATH as it goes."""
    t0 = time.time()
    seen = _existing_ids(CAND_PATH)
    print(f"[generate] target~{target}  existing={len(seen)}  batch={batch_size} workers={workers}")

    combos = _load_combos()
    print(f"[generate] {len(combos)} label combos")

    # --- coverage synthesis ---
    coverage_jobs = [(c, coverage_per) for c in combos]
    n_cov = len(coverage_jobs)
    for start in range(0, n_cov, workers):
        chunk = coverage_jobs[start:start + workers]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_synth_one, *j) for j in chunk]
            rows = []
            for f in as_completed(futs):
                try:
                    rows += f.result()
                except Exception as e:
                    print("  ! synth job failed:", str(e)[:80])
        _append_unique(rows, seen)
        print(f"[generate] coverage {min(start+workers, n_cov)}/{n_cov}  (total {len(seen)})")

    # --- bulk diverse ---
    need = target - len(seen)
    n_batches = max(0, (need + batch_size - 1) // batch_size)
    print(f"[generate] diverse: {n_batches} batches x {batch_size}")
    for bstart in range(0, n_batches, workers):
        chunk = [batch_size] * min(workers, n_batches - bstart)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_diverse_one, n) for n in chunk]
            rows = []
            for f in as_completed(futs):
                try:
                    rows += f.result()
                except Exception as e:
                    print("  ! diverse job failed:", str(e)[:80])
        _append_unique(rows, seen)
        if (bstart // workers) % 10 == 0:
            print(f"[generate] diverse {min(bstart+workers, n_batches)}/{n_batches}  (total {len(seen)}, {time.time()-t0:.0f}s)")

    print(f"[generate] {len(seen)} unique candidates -> {CAND_PATH}  ({time.time()-t0:.0f}s)")
    return len(seen)


def _append_unique(rows: list[dict], seen: set[str]) -> None:
    fresh = []
    for c in rows:
        c["text"] = str(c.get("text", "")).strip()
        if not c["text"]:
            continue
        h = _sha(c["text"])
        if h in seen:
            continue
        seen.add(h)
        c["id"] = h[:16]
        c["hint"] = list(c.get("hint") or [])
        fresh.append(c)
    if fresh:
        _append_jsonl(CAND_PATH, fresh)


# --------------------------------------------------------------------------- #
# Phase 2 — Jev labeling (chunked, resumable)
# --------------------------------------------------------------------------- #
def _load_candidates() -> list[dict]:
    out = []
    if CAND_PATH.exists():
        with open(CAND_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def _label_chunk(cands: list[dict], threshold: float = 0.4) -> list[dict]:
    q = jev_labels.build_questions()
    texts = [c["text"] for c in cands]
    results = jev_client.batch_systemone(texts, q, max_workers=LABEL_WORKERS)
    rows = []
    for c, res in zip(cands, results):
        rows.append({
            "id": c.get("id") or _sha(c["text"])[:16],
            "text": c["text"],
            "source": c.get("source", "unknown"),
            "labels": jev_labels.hard_labels(res, threshold=threshold),
            "soft": jev_labels.soft_vector(res),
            "model": res.model,
        })
    return rows


def label(chunk_size: int = 400):
    cands = _load_candidates()
    print(f"[label] {len(cands)} candidates")

    # resume: a row counts as "done" only if it actually got labels
    # (rows with empty labels are failed requests — re-label them).
    done_ids: set[str] = set()
    if LABEL_PATH.exists():
        with open(LABEL_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("labels") and len(r["labels"]) > 0:
                    done_ids.add(r["id"])
    todo = [c for c in cands if (c.get("id") or _sha(c["text"])[:16]) not in done_ids]
    print(f"[label] already done={len(done_ids)}  todo={len(todo)}")

    t0 = time.time()
    n = 0
    for i in range(0, len(todo), chunk_size):
        chunk = todo[i : i + chunk_size]
        rows = _label_chunk(chunk)
        _append_jsonl(LABEL_PATH, rows)
        n += len(rows)
        el = time.time() - t0
        rate = n / el if el > 0 else 0
        print(f"[label] {n}/{len(todo)}  ({rate:.1f}/s, elapsed {el:.0f}s)")
    print(f"[label] done -> {LABEL_PATH} ({time.time()-t0:.0f}s)")


# --------------------------------------------------------------------------- #
# Phase 3 — finalize
# --------------------------------------------------------------------------- #
def finalize(seed: int = 42):
    rows = []
    with open(LABEL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="id").reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_val = max(1, int(n * 0.05))
    n_test = max(1, int(n * 0.05))
    split = ["train"] * (n - n_val - n_test) + ["val"] * n_val + ["test"] * n_test
    df["split"] = split
    df.to_parquet(OUT_PATH, index=False)
    print(f"[finalize] {n} rows -> {OUT_PATH}  train={n-n_val-n_test} val={n_val} test={n_test}")
    risk = df["labels"].apply(lambda L: next((l for l in L if l.startswith("risk.")), None))
    print("risk dist:", risk.value_counts().to_dict())
    return df


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "generate"
    if cmd == "generate":
        target = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
        generate(target)
    elif cmd == "label":
        label()
    elif cmd == "finalize":
        finalize()
    else:
        print(__doc__)
