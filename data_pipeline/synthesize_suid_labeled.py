"""Synthesize SUID-escalation interpreter-escape commands WITH hard labels in one call.

The LLM emits the command AND its v2 hard labels (verdict + actions) together,
so we don't waste a second Jev labeling pass. Jev is used only for spot-check QA.

Model: deepseek-v4-flash (the only gateway model that does NOT redact shell-escape
payloads). gpt-5.6-sol redacts `/bin/sh` into placeholders -> unusable for this.

Output JSONL: {text, verdict, actions, source}
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from data_pipeline import llm
import schema

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset" / "suid_synth_labeled.jsonl"

ACTION_IDS = list(schema.action_ids())
VERDICTS = list(schema.verdict_ids())

SYSTEM = (
    "You are a defensive security researcher building a labeled intrusion-detection "
    "test dataset for a Linux shell-command threat classifier. For each command you "
    "generate, also provide its ground-truth labels. Output ONLY JSON."
)

USER_TPL = (
    "Generate {n} DIVERSE Linux commands that abuse a legitimate tool's eval/exec/! hook "
    "to escape a SUID binary and spawn a shell (the classic GTFOBins technique).\n"
    "Style reference (vary the tool and syntax, do NOT copy):\n"
    "  - awk 'BEGIN {{system(\"/bin/sh\")}}'\n"
    "  - find . -exec /bin/sh -p \\; -quit\n"
    "  - make --eval='$(shell /bin/sh)' .\n"
    "  - git -c core.pager='sh -c sh' log\n"
    "  - vim -c ':!/bin/sh'\n\n"
    "For each command pick ONE verdict from: benign, suspicious, malicious "
    "(these are clearly offensive -> malicious).\n"
    "Pick applicable actions from this closed set: {actions}.\n"
    "Output JSON: {{\"samples\":[{{\"cmd\":\"...\",\"verdict\":\"...\",\"actions\":[\"...\"]}}]}}"
)


def build_prompt(n: int) -> str:
    return USER_TPL.format(n=n, actions=", ".join(a.removeprefix("action.") for a in ACTION_IDS))


def parse(obj) -> list[dict]:
    out = []
    for s in (obj or {}).get("samples") or []:
        if not isinstance(s, dict):
            continue
        cmd = (s.get("cmd") or "").strip()
        verdict = (s.get("verdict") or "").strip()
        actions = [a.strip() for a in (s.get("actions") or []) if isinstance(a, str)]
        if not cmd or verdict not in ("benign", "suspicious", "malicious"):
            continue
        if not re.search(r'/bin/sh|/bin/bash|\bsh\b|bash|system\(|exec|!/bin/sh|--eval|--exec|shell', cmd, re.I):
            continue
        full_actions = [f"action.{a}" for a in actions if f"action.{a}" in schema.action_ids()]
        out.append({"text": cmd, "verdict": f"verdict.{verdict}", "actions": full_actions})
    return out


def gen_chunk(n: int, model: str, temperature: float) -> list[dict]:
    for _ in range(4):
        try:
            obj, res = llm.chat_json(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": build_prompt(n)}],
                model=model, temperature=temperature, max_tokens=4096, timeout=120.0, retries=3,
            )
        except Exception:
            continue
        rows = parse(obj)
        if rows:
            return rows
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--model", type=str, default="deepseek-v4-flash")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                seen.add(json.loads(line)["text"])
            except Exception:
                pass

    written = 0
    it = 0
    while written < args.n and it < 40:
        it += 1
        need = min(args.chunk, args.n - written)
        rows = gen_chunk(need, args.model, args.temperature)
        fresh = 0
        with open(out, "a") as f:
            for r in rows:
                if r["text"] in seen:
                    continue
                seen.add(r["text"])
                r["source"] = "llm_suid_labeled"
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                written += 1
                fresh += 1
        print(f"  iter {it}: +{fresh} (total {written}/{args.n})", flush=True)
        if fresh == 0:
            time.sleep(1)
    print(f"done -> {out} ({written})")


if __name__ == "__main__":
    import time
    main()
