"""Generate balanced suspicious/benign candidates and append to the candidate pool.

Resumable: skips batches already present by hash. Appends to candidates_50k.jsonl
so the label phase picks them up in a later pass.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hyperids import llm, prompts

ROOT = Path(__file__).resolve().parents[2]
CAND = ROOT / "dataset" / "candidates_50k.jsonl"
MODEL = "feature/flash"
WORKERS = 8
BATCH = 30


def _sha(t: str) -> str:
    return hashlib.sha1(t.strip().encode()).hexdigest()


def _existing() -> set[str]:
    seen = set()
    if CAND.exists():
        for line in open(CAND):
            line = line.strip()
            if line:
                try:
                    seen.add(json.loads(line)["id"])
                except Exception:
                    pass
    return seen


def _gen(prompt_fn, n):
    parsed, _ = llm.chat_json(prompt_fn(n), model=MODEL, temperature=1.0, max_tokens=8192)
    return [str(s).strip() for s in (parsed or {}).get("commands") or [] if str(s).strip()]


def run(suspicious: int = 15000, benign: int = 15000):
    seen = _existing()
    t0 = time.time()
    for name, prompt_fn, target in (("suspicious", prompts.suspicious_prompt, suspicious),
                                    ("benign", prompts.benign_prompt, benign)):
        need = target
        batches = 0
        while need > 0:
            chunk = [BATCH] * min(WORKERS, max(1, (need + BATCH - 1) // BATCH))
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futs = [ex.submit(_gen, prompt_fn, n) for n in chunk]
                rows = []
                for f in as_completed(futs):
                    try:
                        for t in f.result():
                            h = _sha(t)
                            if h[:16] in seen:
                                continue
                            seen.add(h[:16])
                            rows.append({"id": h[:16], "text": t, "source": name, "hint": []})
                    except Exception as e:
                        print(f"  ! {name} batch failed: {str(e)[:70]}", flush=True)
            with open(CAND, "a") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            need -= len(rows)
            batches += 1
            if batches % 20 == 0:
                print(f"[{name}] added ~{target-need}/{target}  total={len(seen)}  ({time.time()-t0:.0f}s)", flush=True)
        print(f"[{name}] done: +{target-need}  total={len(seen)}", flush=True)
    print(f"[balanced] total candidates now {len(seen)}", flush=True)


if __name__ == "__main__":
    sus = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
    ben = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
    run(sus, ben)
