"""Stage 3 — synthesize new commands conditioned on canonical label sets.

Uses the FAST model by default (qwen-flash): speed > precision for bulk
variants; the strong model is reserved for hard negatives / red team.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from data_pipeline import llm, normalize, prompts
from data_pipeline import Sample

SEED_LABELED = Path(__file__).resolve().parents[1] / "dataset" / "seed_labeled.parquet"
OUT_PATH = Path(__file__).resolve().parents[1] / "dataset" / "synthetic.parquet"


def generate(labels: list[str], n: int, model: str = llm.FAST_MODEL, temperature: float = 0.9) -> list[str]:
    """Return n synthetic commands matching `labels`."""
    msgs = prompts.synthesize_prompt(labels, n=n)
    parsed, _ = llm.chat_json(msgs, model=model, temperature=temperature, max_tokens=2048)
    samples = (parsed or {}).get("samples") or []
    return [str(s).strip() for s in samples if str(s).strip()]


def run(
    n_per_combo: int = 4,
    n_combos: int = 8,
    model: str = llm.FAST_MODEL,
    seed: int = 42,
    max_workers: int = 6,
) -> pd.DataFrame:
    rng = random.Random(seed)
    base = pd.read_parquet(SEED_LABELED)
    # build label combos from non-empty labeled rows
    combos = [r["labels"] for _, r in base.iterrows() if len(r["labels"]) > 0]
    rng.shuffle(combos)
    combos = combos[:n_combos]

    rows: list[Sample] = []
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor, as_completed

    jobs = [(c, i, n_per_combo, model) for i, c in enumerate(combos)]
    with ThreadPoolExecutor(max_workers=min(max_workers, 8)) as ex:
        futs = {ex.submit(_gen_one, *j): j for j in jobs}
        for fut in as_completed(futs):
            try:
                rows.extend(fut.result())
            except Exception as e:
                print("gen error:", str(e)[:80])

    df = pd.DataFrame([asdict(s) for s in rows])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"generated {len(df)} synthetic samples from {len(combos)} combos  ({time.time()-t0:.1f}s)")
    print(f"→ {OUT_PATH}")
    return df


def _gen_one(labels: list[str], combo_idx: int, n: int, model: str) -> list[Sample]:
    cmds = generate(labels, n, model=model)
    out = []
    for j, c in enumerate(cmds):
        out.append(Sample(
            id=f"syn-{combo_idx:03d}-{j:02d}",
            text=c,
            labels=list(labels),
            source="synthetic",
            parent=None,
            meta={"model": model, "prompt_labels": list(labels)},
        ))
    return out


if __name__ == "__main__":
    import sys
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    n_combo = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    model = sys.argv[3] if len(sys.argv) > 3 else llm.FAST_MODEL
    df = run(n_per_combo=n_per, n_combos=n_combo, model=model)
    print(df[["id", "text", "labels"]].head(12).to_string(max_colwidth=80))
