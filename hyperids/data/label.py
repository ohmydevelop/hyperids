"""Stage 2 — label real seed commands with the Frontier LLM.

For each seed: LLM assigns risk + intents + tactics + techniques, output is
normalized to canonical schema ids and validated. Writes parquet to dataset/.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from hyperids import llm, normalize, prompts, seeds
from hyperids import Sample

OUT_PATH = Path(__file__).resolve().parents[2] / "dataset" / "seed_labeled.parquet"


def _label_one(seed: dict, model: str) -> Sample | None:
    cmd = seed["text"]
    for attempt in range(3):
        parsed, res = llm.chat_json(prompts.label_prompt(cmd), model=model)
        if parsed is None:
            continue
        labels = normalize.combined_labels(parsed)
        # risk is mandatory — every row must have exactly one risk label
        if labels and any(x.startswith("risk.") for x in labels):
            return Sample(
                id=seed["id"],
                text=cmd,
                labels=labels,
                source="seed",
                meta={
                    "model": model,
                    "latency_s": round(res.latency_s, 3),
                    "seed_risk": seed["seed_risk"],
                    "raw": parsed,
                },
            )
    # fallback: keep a record but mark as unlabeled
    return Sample(
        id=seed["id"], text=cmd, labels=[], source="seed",
        meta={"model": model, "seed_risk": seed["seed_risk"], "error": "failed_to_label"},
    )


def run(model: str = llm.DEFAULT_MODEL, max_workers: int = 6, limit: int | None = None) -> pd.DataFrame:
    rows = seeds.ALL_SEEDS[:limit] if limit else seeds.ALL_SEEDS
    t0 = time.time()
    out: list[Sample] = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=min(max_workers, 8)) as ex:
        futs = {ex.submit(_label_one, s, model): i for i, s in enumerate(rows)}
        res = [None] * len(rows)
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                res[i] = fut.result()
            except Exception as e:
                res[i] = Sample(id=rows[i]["id"], text=rows[i]["text"], labels=[],
                                source="seed", meta={"error": str(e)[:100]})

    df = pd.DataFrame([asdict(s) for s in res])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    n = len(df)
    labeled = df["labels"].apply(lambda x: len(x) > 0).sum()
    risk_ok = df.apply(
        lambda r: r["meta"].get("seed_risk") == "malicious"
        and any(l.startswith("risk.malicious") for l in r["labels"])
        or r["meta"].get("seed_risk") == "benign"
        and any(l.startswith("risk.benign") for l in r["labels"]),
        axis=1,
    ).sum()
    print(f"labeled {labeled}/{n} rows  ({time.time()-t0:.1f}s)")
    print(f"risk consistent with seed_risk: {risk_ok}/{n}")
    print(f"→ {OUT_PATH}")
    return df


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else llm.DEFAULT_MODEL
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    df = run(model=model, limit=limit)
    print(df[["id", "text", "labels"]].head(10).to_string())
