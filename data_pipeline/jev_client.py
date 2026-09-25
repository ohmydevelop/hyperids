"""Jev (TypeSafe AI, System One decision model) client.

Jev does NOT use chat/completions — it uses the SystemOne decisions endpoint:
    POST https://api.typesafe.ai/v1/systemone
    {"state": "...", "model": "jev-latest", "questions": {...}}

Question types:
    choice  -> {"type":"choice", "instructions":..., "criteria":{id: desc}}  (<=255 options)
    noul    -> {"type":"noul",  "instructions": "..."}                        (yes/no probability)
    score   -> {"type":"score",  "instructions":..., "criteria":[levels...]}   (<=10 levels)

Key lives in .jev_key (chmod 600), mirroring .key.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_KEY = (Path(__file__).resolve().parents[1] / ".jev_key").read_text().strip()
DEFAULT_MODEL = "jev-latest"


@dataclass
class JevResult:
    model: str
    answers: dict
    usage: dict | None = None
    latency_s: float = 0.0


def systemone(
    state: str,
    questions: dict,
    model: str = DEFAULT_MODEL,
    timeout: float = 30.0,
    retries: int = 3,
) -> JevResult:
    payload = {"state": state, "model": model, "questions": questions}
    data = json.dumps(payload).encode()
    last: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            JEV_URL,
            data=data,
            headers={"Authorization": f"Bearer {JEV_KEY}", "Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            return JevResult(
                model=d.get("model", model),
                answers=d.get("answers", {}),
                usage=d.get("usage"),
                latency_s=time.time() - t0,
            )
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 401 or e.code == 403:
                raise  # bad key — don't retry
            time.sleep(0.6 * (2**attempt))
        except Exception as e:
            last = e
            time.sleep(0.4 * (2**attempt))
    raise RuntimeError(f"Jev call failed after {retries+1} tries: {last}")


def batch_systemone(
    states: list[str],
    questions: dict,
    model: str = DEFAULT_MODEL,
    max_workers: int = 8,
    **kwargs,
) -> list[JevResult]:
    """Concurrent classification of many states against the SAME questions."""
    out: list[JevResult] = [None] * len(states)
    with ThreadPoolExecutor(max_workers=min(max_workers, 24)) as ex:
        futs = {ex.submit(systemone, s, questions, model=model, **kwargs): i for i, s in enumerate(states)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                out[i] = fut.result()
            except Exception as e:
                out[i] = JevResult(model=model, answers={}, latency_s=0.0, usage=None)
    return out
