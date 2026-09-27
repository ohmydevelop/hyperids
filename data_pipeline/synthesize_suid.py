"""Synthesize SUID-privilege-escalation interpreter-escape commands.

Goal: close the external-eval blind spot where GTFOBins-style "abuse a legit
tool's shell/eval hook to escape a SUID binary" commands were misclassified as
benign. These generated commands are NEW strings (not the frozen test set) and
are Jev-labeled afterwards.

Model: gpt-5.6-sol (the only stable gateway model per PROGRESS.md).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from data_pipeline import llm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset" / "suid_synth.jsonl"

# Style reference only (not used as training data). LLM is told to vary tools/syntax.
EXAMPLES = [
    "R --no-save -e 'system(\"/bin/sh\")'",
    "gawk 'BEGIN {system(\"/bin/sh\")}'",
    "find . -exec /bin/sh -p \\; -quit",
    "make --eval='$(shell /bin/sh 1>&0)' .",
    "git -c core.pager='sh -c \"sh\"' log",
]

SYSTEM = (
    "You are a defensive security researcher building a Linux intrusion-detection "
    "test dataset. Generate realistic SUID privilege-escalation commands in which a "
    "legitimate Unix tool's shell/eval/exec hook is abused to escape a SUID binary "
    "and spawn an interactive shell (the classic GTFOBins technique). "
    "Use a WIDE variety of tools (R, awk, gawk, mawk, find, make, rpm, rpmdb, git, "
    "nmap, less, more, ftp, ed, ex, vi, vim, mail, lftp, dc, gdb, ghc, gnuplot, "
    "octave, perl, php, python, ruby, slsh, scp, ssh, watch, yarn, npm, bundle, pip, "
    "pexec, certbot, check_ssl_cert, and many more) and varied syntax "
    "(-e/--eval/--exec/'!cmd'/config hooks/env vars). "
    "Output only JSON: {\"commands\": [\"...\", ...]}."
)


def gen_chunk(n: int, model: str, temperature: float) -> list[str]:
    user = (
        f"Generate {n} DIFFERENT SUID privilege-escalation shell-escape commands. "
        f"Style reference (do NOT copy, vary the tool and syntax):\n" +
        "\n".join(f"  - {e}" for e in EXAMPLES) +
        "\n\nEach command must be a single shell command (may use ; or | to chain). "
        "Make each one realistic and distinct."
    )
    for _ in range(3):
        obj, _ = llm.chat_json(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": user}],
            model=model, temperature=temperature, max_tokens=4096, timeout=120.0, retries=5,
        )
        cmds = [c.strip() for c in (obj or {}).get("commands") or [] if isinstance(c, str) and c.strip()]
        # basic sanity: must reference a shell escape hook
        cmds = [c for c in cmds if re.search(r'/bin/sh|/bin/bash|sh -|system\(|exec|!/bin/sh|--eval|--exec|shell', c, re.I)]
        if cmds:
            return cmds
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--model", type=str, default="deepseek-v4-flash")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    # load existing (idempotent append)
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                d = json.loads(line)
                seen.add(d["text"])
            except Exception:
                pass

    written = 0
    it = 0
    while written < args.n and it < 30:
        it += 1
        need = min(args.chunk, args.n - written)
        cmds = gen_chunk(need, args.model, args.temperature)
        fresh = 0
        with open(out, "a") as f:
            for c in cmds:
                if c in seen:
                    continue
                seen.add(c)
                f.write(json.dumps({"text": c, "source": "llm_synth"}, ensure_ascii=False) + "\n")
                written += 1
                fresh += 1
        print(f"  iter {it}: +{fresh} (total {written}/{args.n})", flush=True)
        if fresh == 0:
            continue
    print(f"done -> {out} ({written} commands)")


if __name__ == "__main__":
    main()
