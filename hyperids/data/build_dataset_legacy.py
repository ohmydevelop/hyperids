"""Dataset builder — LLM proposes candidates, Jev is the sole labeler.

Stages:
  1. synthesize  — new commands per seed label combo (fast: qwen-flash)
  2. obfuscate   — variants of malicious/suspicious seeds (keep behavior)
  3. hard_neg    — near-miss samples (different risk)
  4. min_pair    — benign<->malicious single-edit pairs
  5. Jev soft-label everything (199-dim) + threshold hard labels
  6. dedupe + train/val/test split + write parquet

Authoritative labels always come from Jev; the LLM's proposed labels are only
sampling hints and are not trusted.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from hyperids import jev_client, jev_labels_legacy, llm, prompts
from hyperids import Sample

ROOT = Path(__file__).resolve().parents[2]
SEED_HARD = ROOT / "dataset" / "seed_hard_jev.parquet"
OUT = ROOT / "dataset" / "soft_labels.parquet"


@dataclass
class Candidate:
    text: str
    source: str
    parent: str | None = None
    hint: list[str] = field(default_factory=list)


def _dedupe(cands: list[Candidate]) -> list[Candidate]:
    seen: dict[str, Candidate] = {}
    for c in cands:
        key = hashlib.sha1(c.text.encode()).hexdigest()
        seen.setdefault(key, c)
    return list(seen.values())


def _parse_json_obj(content: str) -> dict | None:
    t = content.strip()
    if t.startswith("```"):
        seg = t.split("```")
        t = seg[1] if len(seg) > 1 else t
        t = t.removeprefix("json").strip()
    i, j = t.find("{"), t.rfind("}")
    if i == -1 or j == -1:
        return None
    try:
        return json.loads(t[i : j + 1])
    except Exception:
        return None


def synthesize_candidates(combos: list[list[str]], per: int, model: str) -> list[Candidate]:
    out: list[Candidate] = []
    for i, labels in enumerate(combos):
        parsed, _ = llm.chat_json(prompts.synthesize_prompt(labels, n=per), model=model, temperature=0.9, max_tokens=2048)
        for s in (parsed or {}).get("samples") or []:
            if str(s).strip():
                out.append(Candidate(str(s).strip(), "synthetic", None, list(labels)))
    return out


def obfuscate_candidates(rows: list[tuple[str, list[str]]], model: str) -> list[Candidate]:
    out: list[Candidate] = []
    for text, labels in rows:
        parsed, _ = llm.chat_json(prompts.obfuscate_prompt(text, labels), model=model, temperature=0.9, max_tokens=2048)
        for v in (parsed or {}).get("variants") or []:
            if str(v).strip():
                out.append(Candidate(str(v).strip(), "obfuscated", None, list(labels)))
    return out


def hard_neg_candidates(rows: list[tuple[str, list[str]]], model: str) -> list[Candidate]:
    out: list[Candidate] = []
    for text, labels in rows:
        parsed, _ = llm.chat_json(prompts.hard_negative_prompt(text, labels), model=model, temperature=0.9, max_tokens=2048)
        for s in (parsed or {}).get("samples") or []:
            cmd = s.get("command") if isinstance(s, dict) else None
            if cmd and str(cmd).strip():
                out.append(Candidate(str(cmd).strip(), "hard_neg", text, []))
    return out


def min_pair_candidates(rows: list[tuple[str, list[str]]], model: str) -> list[Candidate]:
    out: list[Candidate] = []
    for text, labels in rows:
        parsed, _ = llm.chat_json(prompts.minimal_pair_prompt(text, labels), model=model, temperature=0.9, max_tokens=2048)
        for p in (parsed or {}).get("pairs") or []:
            if not isinstance(p, dict):
                continue
            for k in ("benign", "malicious"):
                if str(p.get(k, "")).strip():
                    out.append(Candidate(str(p[k]).strip(), "min_pair", text, []))
    return out


def generate_candidates(target: int = 400, fast_model: str = llm.FAST_MODEL,
                        strong_model: str = llm.DEFAULT_MODEL) -> list[Candidate]:
    seeds = pd.read_parquet(SEED_HARD)
    combos = [list(r["labels"]) for _, r in seeds.iterrows() if len(r["labels"]) > 0]
    malicious = [(r["text"], list(r["labels"])) for _, r in seeds.iterrows()
                 if any(l == "risk.malicious" for l in r["labels"])]
    suspicious = [(r["text"], list(r["labels"])) for _, r in seeds.iterrows()
                  if any(l == "risk.suspicious" for l in r["labels"])]

    n_syn = max(1, target // max(1, len(combos)))
    t0 = time.time()
    print(f"synthesize: {len(combos)} combos x {n_syn} (fast={fast_model}) ...")
    cands = synthesize_candidates(combos, n_syn, fast_model)

    print(f"obfuscate: {len(malicious)} malicious (strong={strong_model}) ...")
    cands += obfuscate_candidates(malicious[:12], strong_model)

    print(f"hard_neg: {len(malicious)} malicious ...")
    cands += hard_neg_candidates(malicious[:8], strong_model)

    print(f"min_pair: {len(suspicious)+len(malicious)} ...")
    cands += min_pair_candidates((suspicious + malicious)[:8], strong_model)

    cands = _dedupe(cands)
    print(f"generated {len(cands)} unique candidates in {time.time()-t0:.0f}s")
    return cands


def jev_label(cands: list[Candidate], max_workers: int = 8, threshold: float = 0.4) -> pd.DataFrame:
    q = jev_labels.build_questions()
    t0 = time.time()
    results = jev_client.batch_systemone([c.text for c in cands], q, max_workers=max_workers)
    rows = []
    for c, res in zip(cands, results):
        rows.append({
            "id": hashlib.sha1(c.text.encode()).hexdigest()[:16],
            "text": c.text,
            "source": c.source,
            "parent": c.parent,
            "labels": jev_labels.hard_labels(res, threshold=threshold),
            "soft": jev_labels.soft_vector(res),
            "model": res.model,
            "latency_s": round(res.latency_s, 3),
        })
    df = pd.DataFrame(rows)
    print(f"Jev labeled {len(df)} candidates in {time.time()-t0:.0f}s")
    return df


def split_write(df: pd.DataFrame, out_path: Path = OUT, seed: int = 42) -> None:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_val = max(1, int(n * 0.1))
    n_test = max(1, int(n * 0.1))
    n_train = n - n_val - n_test
    split = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    df["split"] = split
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"→ {out_path}  train={n_train} val={n_val} test={n_test}")
    # label coverage stats
    all_lab = df["labels"].explode()
    print("label coverage:", all_lab.value_counts().head(10).to_dict())


def run(target: int = 400) -> pd.DataFrame:
    cands = generate_candidates(target)
    df = jev_label(cands)
    split_write(df)
    return df


if __name__ == "__main__":
    import sys
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    run(target)
