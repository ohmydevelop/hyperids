"""红队闭环 — 一轮迭代：找错例 → LLM 造变体 → Jev 重标注 → 合并重训。

Round 1 流程:
    find     — 用当前模型在 test 上找 hard examples（technique 漏标 / risk 判错）
    generate — 针对每个 hard example 的错误标签造变体（synthesize + obfuscate）
    label    — Jev 对变体打 199 维软标签 + hard labels
    merge    — 转成 GLiClass 训练格式，合并进 train.json
    （retrain 用 model/finetune.py，在 GPU 上跑）
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.predict import Predictor
from data_pipeline import llm, prompts, jev_client, jev_labels
from hyperids.schema_legacy import all_label_ids, group_offsets

IDS = all_label_ids()
OFF = group_offsets()
RISK = IDS[OFF["risk"][0]: OFF["risk"][1]]
TECH = IDS[OFF["technique"][0]: OFF["technique"][1]]

TEST_JSON = ROOT / "dataset" / "gliclass" / "test.json"
TRAIN_JSON = ROOT / "dataset" / "gliclass" / "train.json"
HARD_PATH = ROOT / "redteam" / "hard_examples.json"
VAR_PATH = ROOT / "redteam" / "variants_labeled.json"


def _sha(t: str) -> str:
    return hashlib.sha1(t.strip().encode()).hexdigest()


# --------------------------------------------------------------------------- #
# 1. find — hard examples
# --------------------------------------------------------------------------- #
def find_hard(limit: int = 1500, top_n: int = 150):
    pred = Predictor()
    data = json.loads(TEST_JSON.read_text())[:limit]
    hard = []
    for ex in data:
        p = pred.predict(ex["text"])
        pred_risk = p["risk"][0]
        true = set(ex["true_labels"])
        pred_all = set(p["risk"]) | set(p["intent"]) | set(p["tactic"]) | set(p["technique"])
        fn = true - pred_all
        fp = pred_all - true
        true_risk = next((l for l in true if l in RISK), None)
        risk_err = 1 if (true_risk and pred_risk != true_risk) else 0
        tech_fn = len([l for l in fn if l in TECH])
        score = tech_fn * 2 + len(fn) + risk_err * 3 + len(fp)
        if score > 0:
            hard.append({
                "text": ex["text"],
                "true_labels": sorted(true),
                "error_labels": sorted(fn),
                "pred_risk": pred_risk,
                "true_risk": true_risk,
                "score": score,
            })
    hard.sort(key=lambda x: -x["score"])
    hard = hard[:top_n]
    HARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    HARD_PATH.write_text(json.dumps(hard, ensure_ascii=False, indent=2))
    print(f"found {len(hard)} hard examples (from {len(data)} test samples) → {HARD_PATH}")
    # distribution of error labels
    from collections import Counter
    c = Counter(l for h in hard for l in h["error_labels"])
    print("top error labels:", c.most_common(12))
    return hard


# --------------------------------------------------------------------------- #
# 2. generate — LLM variants targeting error labels
# --------------------------------------------------------------------------- #
def _gen_for_hard(h, model, synth_n=5):
    out = []
    # (a) synthesize commands with the MISSED labels (boost recall)
    labels = h["error_labels"] or h["true_labels"]
    p, _ = llm.chat_json(prompts.synthesize_prompt(labels, n=synth_n), model=model, temperature=0.9, max_tokens=4096)
    for s in (p or {}).get("samples") or []:
        if str(s).strip():
            out.append({"text": str(s).strip(), "source": "redteam_synth", "parent": h["text"]})
    # (b) obfuscate the original (robustness)
    p2, _ = llm.chat_json(prompts.obfuscate_prompt(h["text"], h["true_labels"]), model=model, temperature=0.9, max_tokens=4096)
    for v in (p2 or {}).get("variants") or []:
        if str(v).strip():
            out.append({"text": str(v).strip(), "source": "redteam_obf", "parent": h["text"]})
    return out


def generate(model: str = "feature/flash", workers: int = 8, synth_n: int = 5):
    hard = json.loads(HARD_PATH.read_text())
    cands: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_gen_for_hard, h, model, synth_n) for h in hard]
        for f in as_completed(futs):
            try:
                cands += f.result()
            except Exception as e:
                print("  ! gen failed:", str(e)[:70])
    # dedupe
    seen, uniq = set(), []
    for c in cands:
        k = _sha(c["text"])
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    print(f"generated {len(uniq)} unique variants in {time.time()-t0:.0f}s")
    return uniq


# --------------------------------------------------------------------------- #
# 3. label — Jev soft labels
# --------------------------------------------------------------------------- #
def label(cands, max_workers=12):
    q = jev_labels.build_questions()
    results = jev_client.batch_systemone([c["text"] for c in cands], q, max_workers=max_workers)
    rows = []
    for c, r in zip(cands, results):
        rows.append({
            "id": _sha(c["text"])[:16],
            "text": c["text"],
            "source": c["source"],
            "parent": c["parent"],
            "labels": jev_labels.hard_labels(r, threshold=0.4),
            "soft": jev_labels.soft_vector(r),
        })
    VAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    VAR_PATH.write_text(json.dumps(rows, ensure_ascii=False))
    print(f"Jev labeled {len(rows)} variants → {VAR_PATH}")
    return rows


# --------------------------------------------------------------------------- #
# 4. merge — into GLiClass training JSON
# --------------------------------------------------------------------------- #
def merge(neg_k=20):
    rows = json.loads(VAR_PATH.read_text())
    train = json.loads(TRAIN_JSON.read_text())
    existing = {_sha(d["text"]) for d in train}
    import random
    rng = random.Random(42)
    added = 0
    for r in rows:
        if not r.get("labels"):
            continue
        if _sha(r["text"]) in existing:
            continue
        true = [l for l in r["labels"] if l in set(IDS)]
        if not true:
            continue
        negatives = [l for l in RISK if l not in true]
        pool = [l for l in IDS if l not in true and l not in negatives]
        rng.shuffle(pool)
        negatives += pool[:neg_k]
        train.append({"text": r["text"], "true_labels": true, "all_labels": true + negatives})
        existing.add(_sha(r["text"]))
        added += 1
    TRAIN_JSON.write_text(json.dumps(train, ensure_ascii=False))
    print(f"merged +{added} samples → {TRAIN_JSON} (train total={len(train)})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["find", "generate", "label", "merge", "all"])
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--top_n", type=int, default=150)
    ap.add_argument("--model", type=str, default="feature/flash")
    args = ap.parse_args()

    if args.step == "find":
        find_hard(limit=args.limit, top_n=args.top_n)
    elif args.step == "generate":
        cands = generate(model=args.model)
        json.dump(cands, open(ROOT / "redteam" / "candidates.json", "w"), ensure_ascii=False)
    elif args.step == "label":
        cands = json.loads((ROOT / "redteam" / "candidates.json").read_text())
        label(cands)
    elif args.step == "merge":
        merge()
    elif args.step == "all":
        find_hard(limit=args.limit, top_n=args.top_n)
        cands = generate(model=args.model)
        json.dump(cands, open(ROOT / "redteam" / "candidates.json", "w"), ensure_ascii=False)
        label(cands)
        merge()
