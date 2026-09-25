"""Frontier LLM client — thin wrapper over the OpenAI-compatible gateway.

Design constraints (from configs/model_selection.md):
  * concurrency <= 8 (gateway 502s under load)
  * timeout 30s, 5xx exponential backoff x3
  * response_format={"type":"json_object"} for stable JSON
  * temperature=0 by default (diversity handled by callers)
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BASE = "https://ai-api-gateway.app.baizhi.cloud/api/openai/chat/completions"
KEY = (Path(__file__).resolve().parents[1] / ".key").read_text().strip()

DEFAULT_MODEL = "gpt-5.6-sol"
FAST_MODEL = "qwen-flash"
STRONG_MODEL = "deepseek-v4-pro"


@dataclass
class LLMResult:
    model: str
    content: str
    latency_s: float
    usage: dict | None = None


def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    json_mode: bool = True,
    timeout: float = 30.0,
    retries: int = 3,
) -> LLMResult:
    """Single chat completion with retry on transient failures."""
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    data = json.dumps(body).encode()
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            BASE,
            data=data,
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read())
            dt = time.time() - t0
            content = (resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            return LLMResult(model=model, content=content, latency_s=dt, usage=resp.get("usage"))
        except urllib.error.HTTPError as e:
            last_err = e
            # 4xx (invalid model etc.) won't fix itself — fail fast
            if e.code < 500:
                raise
            time.sleep(1.0 * (2**attempt))
        except Exception as e:  # timeouts, conn errors
            last_err = e
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"LLM call failed after {retries+1} tries: {last_err}")


def chat_json(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    **kwargs,
) -> tuple[dict | None, LLMResult]:
    """chat() + parse JSON object; returns (parsed_or_None, result)."""
    res = chat(messages, model=model, **kwargs)
    return _parse_json(res.content), res


def _parse_json(text: str) -> dict | None:
    t = text.strip()
    if t.startswith("```"):
        segs = t.split("```")
        t = segs[1] if len(segs) > 1 else t
        t = t.removeprefix("json").strip()
    i, j = t.find("{"), t.rfind("}")
    if i == -1 or j == -1:
        return None
    try:
        obj = json.loads(t[i : j + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def batch_chat_json(
    prompts: list[list[dict]],
    model: str = DEFAULT_MODEL,
    max_workers: int = 6,
    **kwargs,
) -> list[tuple[dict | None, LLMResult]]:
    """Concurrent batch of chat_json calls; order preserved."""
    from concurrent.futures import ThreadPoolExecutor

    out: list[tuple[dict | None, LLMResult]] = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=min(max_workers, 8)) as ex:
        futs = {ex.submit(chat_json, p, model=model, **kwargs): i for i, p in enumerate(prompts)}
        from concurrent.futures import as_completed

        for fut in as_completed(futs):
            i = futs[fut]
            try:
                out[i] = fut.result()
            except Exception as e:
                out[i] = (None, LLMResult(model=model, content="", latency_s=0.0))
    return out
