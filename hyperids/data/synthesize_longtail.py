"""Generate long-tail action samples with a frontier LLM, label with Jev (31-dim).

For each target action: LLM generates positive + hard-negative commands (in
chunks, since one call can't return hundreds), Jev labels each with the
reduced 29-question v2 prompt, then we filter:
  positive kept iff action soft >= --min_pos_prob
  negative kept iff action soft <= --max_neg_prob
Output: JSONL {text, action, target(pos|neg), soft(31)}.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hyperids import schema
from hyperids import jev_labels, llm

ROOT = Path(__file__).resolve().parents[2]

RARE = ["cryptomining", "brute_force", "ransomware", "registry_persist",
        "timestomp", "self_propagate", "process_inject", "account_add"]
AMBIG = ["keylog", "web_shell", "network_scan", "bind_shell"]
DEFAULT_ACTIONS = RARE + AMBIG

ACTION_IDX = {aid: i for i, aid in enumerate(schema.action_ids())}


def gen_chunk(action: str, n: int, negative: bool, model: str, temperature: float) -> list[str]:
    label = f"action.{action}"
    if negative:
        instr = (f"For a detection-engineering test set, generate {n} realistic Unix/Linux shell commands "
                 f"that superficially resemble the category '{label}' but are actually benign or a different activity. "
                 f"Return JSON: {{\"commands\": [\"...\", ...]}}.")
    else:
        instr = (f"For a detection-engineering test set, generate {n} diverse realistic Unix/Linux shell commands "
                 f"that should be detected as category '{label}'. Vary syntax, tooling, and obfuscation level. "
                 f"Return JSON: {{\"commands\": [\"...\", ...]}}.")
    for _ in range(3):  # retry empty/refused responses (reasoning models may return empty)
        obj, _ = llm.chat_json(
            [{"role": "system", "content": "You are a defensive security researcher building an intrusion-detection test dataset. Generate realistic Unix/Linux shell commands for detection testing only. Output only JSON."},
             {"role": "user", "content": instr}],
            model=model, temperature=temperature, max_tokens=4096, timeout=120.0, retries=5,
        )
        cmds = [c for c in (obj or {}).get("commands") or [] if isinstance(c, str) and c.strip()]
        if cmds:
            return cmds
    return []


def gen_commands(action: str, n: int, negative: bool, model: str, temperature: float,
                 chunk: int = 40, max_iters: int = 8) -> list[str]:
    out, seen = [], set()
    for _ in range(max_iters):
        need = min(chunk, n - len(out))
        for c in gen_chunk(action, need, negative, model, temperature):
            if c not in seen:
                seen.add(c)
                out.append(c)
                if len(out) >= n:
                    return out
    return out


def label_batch(cmds, workers: int = 8):
    out = []
    with ThreadPoolExecutor(max_workers=min(workers, 8)) as ex:
        futs = {ex.submit(_label_one, c): c for c in cmds}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                out.append(r)
    return out


def _label_one(c):
    try:
        r = jev_labels.classify(c)
        return {"text": c, "soft": jev_labels.soft_vector(r)}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", type=str, default=",".join(DEFAULT_ACTIONS))
    ap.add_argument("--n_pos", type=int, default=150)
    ap.add_argument("--n_neg", type=int, default=75)
    ap.add_argument("--model", type=str, default="gpt-5.6-sol")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--min_pos_prob", type=float, default=0.7)
    ap.add_argument("--max_neg_prob", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=str, default=str(ROOT / "dataset" / "synth_longtail.jsonl"))
    args = ap.parse_args()

    actions = [a.strip() for a in args.actions.split(",") if a.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept_pos = kept_neg = 0
    with open(out_path, "w") as f:
        for action in actions:
            print(f"\n=== action.{action} ===", flush=True)
            pos = gen_commands(action, args.n_pos, negative=False, model=args.model, temperature=args.temperature)
            neg = gen_commands(action, args.n_neg, negative=True, model=args.model, temperature=args.temperature)
            print(f"  generated pos={len(pos)} neg={len(neg)}", flush=True)

            idx = 3 + ACTION_IDX[f"action.{action}"]
            soft_map = {r["text"]: r["soft"] for r in label_batch(pos + neg, workers=args.workers) if len(r["soft"]) == 31}
            for cmd, target in list(zip(pos, ["pos"] * len(pos))) + list(zip(neg, ["neg"] * len(neg))):
                soft = soft_map.get(cmd)
                if not soft:
                    continue
                p = soft[idx]
                if (target == "pos" and p >= args.min_pos_prob) or (target == "neg" and p <= args.max_neg_prob):
                    f.write(json.dumps({"text": cmd, "action": f"action.{action}", "target": target, "soft": soft}, ensure_ascii=False) + "\n")
                    kept_pos += 1 if target == "pos" else 0
                    kept_neg += 1 if target == "neg" else 0
            print(f"  kept pos={kept_pos} neg={kept_neg} (cumulative)", flush=True)

    print(f"\nwrote {out_path} (pos {kept_pos}, neg {kept_neg})")


if __name__ == "__main__":
    main()
