"""Jev Teacher — produce soft-label vectors (and hard labels) for commands.

This is the first Teacher artifact: a 199-dim probability vector per command,
in schema.all_label_ids() order, ready to feed Student KD. Also writes a
thresholded hard-label version for QA / baseline hard-label training.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from data_pipeline import jev_client, jev_labels
from data_pipeline.seeds import ALL_SEEDS

OUT_SOFT = Path(__file__).resolve().parents[1] / "dataset" / "seed_soft_jev.parquet"
OUT_HARD = Path(__file__).resolve().parents[1] / "dataset" / "seed_hard_jev.parquet"


def run(commands: list[str] | None = None, ids: list[str] | None = None,
        max_workers: int = 8, threshold: float = 0.4) -> pd.DataFrame:
    if commands is None:
        commands = [s["text"] for s in ALL_SEEDS]
        ids = [s["id"] for s in ALL_SEEDS]

    q = jev_labels.build_questions()
    t0 = time.time()
    results = jev_client.batch_systemone(commands, q, max_workers=max_workers)
    dt = time.time() - t0

    rows = []
    for sid, cmd, res in zip(ids, commands, results):
        vec = jev_labels.soft_vector(res)
        hard = jev_labels.hard_labels(res, threshold=threshold)
        rows.append({
            "id": sid,
            "text": cmd,
            "labels": hard,
            "soft": vec,
            "model": res.model,
            "latency_s": round(res.latency_s, 3),
        })

    df = pd.DataFrame(rows)
    OUT_SOFT.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["labels"]).to_parquet(OUT_SOFT, index=False)  # soft vectors only
    df.drop(columns=["soft"]).to_parquet(OUT_HARD, index=False)    # hard labels only

    n = len(df)
    avg_lat = df["latency_s"].mean()
    n_labels = df["labels"].apply(len)
    print(f"Jev labeled {n}/{len(commands)} commands in {dt:.1f}s "
          f"(avg {avg_lat:.2f}s/req, labels avg {n_labels.mean():.1f}/cmd)")
    print(f"→ {OUT_SOFT}")
    print(f"→ {OUT_HARD}")
    return df


if __name__ == "__main__":
    df = run()
    print(df[["id", "text", "labels"]].head(12).to_string(max_colwidth=60))
