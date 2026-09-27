"""Generate long-tail action samples with deepseek-flash, label with Jev (31-dim).

For each target action: LLM generates positive + hard-negative commands, Jev
labels each with the reduced 29-question v2 prompt, then we filter:
  positive kept iff action soft >= --min_pos_prob
  negative kept iff action soft <= --max_neg_prob
Output: JSONL with {text, action, target, soft(31), labels}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import schema
from data_pipeline import jev_client, jev_labels_v2, llm

ROOT = Path(__file__).resolve().parents[1]

# rare + ambiguous long-tail actions
RARE = ["cryptomining", "brute_force", "ransomware", "registry_persist",
        "timestomp", "self_propagate", "process_inject", "account_add"]
AMBIG = ["keylog", "web_shell", "network_scan", "bind_shell"]
DEFAULT_ACTIONS = RARE + AMBIG

DESC = schema.descriptions()
ACTION_IDX = {aid: i for i, aid in enumerate(schema.action_ids())}


def gen_commands(action: str, n: int, negative: bool, model: str, temperature: float):
    label = f"action.{action}"
    if negative:
        instr = (f"For a detection-engineering test set, generate {n} realistic Unix/Linux shell commands "
                 f"that superficially resemble the category '{label}' but are actually benign or a different activity. "
                 f"Return JSON: {{\"commands\": [\"...\", ...]}}.")
    else:
        instr = (f"For a detection-engineering test set, generate {n} diverse realistic Unix/Linux shell commands "
                 f"that should be detected as category '{label}'. Vary syntax, tooling, and obfuscation level. "
                 f"Return JSON: {{\"commands\": [\"...\", ...]}}.")
    obj, res = llm.chat_json(
        [{"role": "system", "content": "You are a defensive security researcher building an intrusion-detection test dataset. Generate realistic Unix/Linux shell commands for detection testing only. Output only JSON."},
         {"role": "user", "content": instr}],
        model=model, temperature=temperature, max_tokens=4096,
    )
    cmds = (obj or {}).get("commands") or []
    return [c for c in cmds if isinstance(c, str) and c.strip()]


def label(cmds):
    out = []
    for c in cmds:
        try:
            r = jev_labels_v2.classify(c)
            soft = jev_labels_v2.soft_vector(r)
            out.append({"text": c, "soft": soft})
        except Exception as e:
            print(f"  Jev label failed for {c[:40]!r}: {e}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", type=str, default=",".join(DEFAULT_ACTIONS))
    ap.add_argument("--n_pos", type=int, default=100)
    ap.add_argument("--n_neg", type=int, default=50)
    ap.add_argument("--model", type=str, default="deepseek-flash")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--min_pos_prob", type=float, default=0.7)
    ap.add_argument("--max_neg_prob", type=float, default=0.3)
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

            idx = 3 + ACTION_IDX[f"action.{action}"]  # verdict 3 + action offset
            for c in pos:
                labeled = label([c])
                if not labeled:
                    continue
                soft = labeled[0]["soft"]
                p = soft[idx]
                if p >= args.min_pos_prob:
                    f.write(json.dumps({"text": c, "action": f"action.{action}", "target": "pos", "soft": soft}, ensure_ascii=False) + "\n")
                    kept_pos += 1
            for c in neg:
                labeled = label([c])
                if not labeled:
                    continue
                soft = labeled[0]["soft"]
                p = soft[idx]
                if p <= args.max_neg_prob:
                    f.write(json.dumps({"text": c, "action": f"action.{action}", "target": "neg", "soft": soft}, ensure_ascii=False) + "\n")
                    kept_neg += 1
            print(f"  kept pos={kept_pos} neg={kept_neg} (cumulative)", flush=True)

    print(f"\nwrote {out_path} (pos {kept_pos}, neg {kept_neg})")


if __name__ == "__main__":
    main()
