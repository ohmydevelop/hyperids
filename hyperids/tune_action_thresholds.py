"""Tune per-action decision thresholds on the validation set.

Sweeps each action's threshold (0.30..0.70, step 0.05) greedily to maximize
overall action micro-F1, then writes configs/action_thresholds.json.
verdict stays argmax (mutually exclusive).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from gliclass import GLiClassModel
from transformers import AutoTokenizer
from hyperids import schema

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model" / "checkpoints_gpu" / "final_model_v4"
DATA_DIR = ROOT / "dataset" / "gliclass_v2"
OUT = ROOT / "configs" / "action_thresholds.json"

IDS = schema.all_label_ids()
OFF = schema.group_offsets()
ACTIONS = IDS[OFF["action"][0]: OFF["action"][1]]


def score_all(model, tok, cmd, device):
    s = "".join(f"<<LABEL>>{l}" for l in IDS) + "<<SEP>>" + cmd
    enc = tok(s, return_tensors="pt", truncation=True, max_length=320).to(device)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_num_classes=len(IDS))
    lg = out.logits.flatten()
    return {l: float(lg[j].item()) for j, l in enumerate(IDS)}


def micro_f1(thresholds, Y, S):
    tp = fp = fn = 0
    for true, sc in zip(Y, S):
        pred = {a for a in ACTIONS if sc.get(a, -99.0) >= thresholds.get(a, 0.5)}
        true = set(true)
        tp += len(true & pred); fp += len(pred - true); fn += len(true - pred)
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=str(MODEL_DIR))
    ap.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiClassModel.from_pretrained(args.model_dir).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.model_dir, add_prefix_space=True)

    data = json.loads((Path(args.data_dir) / "val.json").read_text())
    if args.limit:
        data = data[: args.limit]
    print(f"tuning thresholds on {len(data)} val samples ({device}) ...", flush=True)

    Y, S = [], []
    for ex in data:
        sc = score_all(model, tok, ex["text"], device)
        S.append(sc)
        Y.append([a for a in ACTIONS if a in ex["true_labels"]])

    thresholds = {a: 0.5 for a in ACTIONS}
    grid = [round(0.30 + 0.05 * i, 2) for i in range(9)]
    for _pass in range(2):
        for a in ACTIONS:
            best_t, best_f1 = thresholds[a], micro_f1(thresholds, Y, S)
            for t in grid:
                thresholds[a] = t
                f1 = micro_f1(thresholds, Y, S)
                if f1 > best_f1:
                    best_t, best_f1 = t, f1
            thresholds[a] = best_t

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(thresholds, indent=2, ensure_ascii=False))
    print(f"action thresholds -> {args.out}")
    print("overall action micro-F1 @ tuned:", round(micro_f1(thresholds, Y, S), 4))


if __name__ == "__main__":
    main()
