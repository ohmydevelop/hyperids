#!/usr/bin/env python3
"""用最强模型（deepseek-v4-pro）仲裁 gt 与 Jev verdict 不一致的红队样本。

只读取 adversarial/labeled.jsonl（冻结），不读 train/val/test。不执行任何命令。
输出：每行 {text, gt, jev_verdict, arb_verdict, arb_reason}
"""
from __future__ import annotations
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hyperids import llm

ROOT = Path(__file__).resolve().parents[1]
LABELED = ROOT / "adversarial" / "labeled.jsonl"
RUN = ROOT / "adversarial" / "runs" / "20260930-v5-hardneg-audit"
OUT = RUN / "arbitration_deepseek_v4_pro.jsonl"
MODEL = "deepseek-v4-pro"

SYS = (
    "You are a defensive security researcher. Classify the security risk level of this shell command "
    "for an intrusion-detection test dataset. Respond with JSON only."
)


def arb_one(text: str) -> dict:
    user = (
        f"Classify this shell command into one of: benign, suspicious, malicious.\n"
        f"Command: {text}\n"
        f'Return JSON: {{"verdict": "benign|suspicious|malicious", "reason": "one short sentence"}}'
    )
    for _ in range(3):
        obj, _ = llm.chat_json(
            [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            model=MODEL, temperature=0.0, max_tokens=256, timeout=60.0, retries=2,
        )
        v = (obj or {}).get("verdict")
        if v in ("benign", "suspicious", "malicious"):
            return {"verdict": v, "reason": (obj or {}).get("reason", "")}
    return {"verdict": "unknown", "reason": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    rows = [json.loads(l) for l in LABELED.read_text().splitlines() if l.strip()]
    amb = [r for r in rows if r.get("gt") in ("benign", "suspicious", "malicious")
           and r.get("verdict") in ("verdict.benign", "verdict.suspicious", "verdict.malicious")
           and r["gt"] != r["verdict"].removeprefix("verdict.")]
    if args.limit:
        amb = amb[: args.limit]
    print(f"ambiguous (gt != jev verdict): {len(amb)}", flush=True)

    results = [None] * len(amb)
    with ThreadPoolExecutor(max_workers=min(args.workers, 8)) as ex:
        futs = {ex.submit(arb_one, r["text"]): i for i, r in enumerate(amb)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = {"verdict": "error", "reason": repr(e)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from collections import Counter
    stat = Counter()
    with out.open("w") as f:
        for r, a in zip(amb, results):
            row = {
                "text": r["text"],
                "gt": r["gt"],
                "jev_verdict": r["verdict"],
                "arb_verdict": a["verdict"],
                "arb_reason": a.get("reason", ""),
                "arbitrator_model": args.model,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stat[(r["gt"], r["verdict"], a["verdict"])] += 1

    print("arbitration done:", dict(stat))
    # 关键口径：gt=malicious 的样本被仲裁为什么
    gm = Counter()
    for r, a in zip(amb, results):
        if r["gt"] == "malicious":
            gm[a["verdict"]] += 1
    print("gt=malicious -> arbitrator verdict:", dict(gm))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
