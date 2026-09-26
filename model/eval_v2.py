"""Evaluate the v2 model (verdict 3 + action 28 = 31 labels).

verdict: argmax (mutually exclusive). action: threshold (single, default 0.5).
MITRE tactic/technique are derived via schema_v2.derive_attck(), not evaluated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from gliclass import GLiClassModel
from transformers import AutoTokenizer
import schema_v2

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model" / "checkpoints_gpu" / "final_model_v2"
DATA_DIR = ROOT / "dataset" / "gliclass_v2"

IDS = schema_v2.all_label_ids()
OFF = schema_v2.group_offsets()
VERDICT = IDS[OFF["verdict"][0]: OFF["verdict"][1]]
ACTIONS = IDS[OFF["action"][0]: OFF["action"][1]]


def score_all(model, tok, cmd, device):
    s = "".join(f"<<LABEL>>{l}" for l in IDS) + "<<SEP>>" + cmd
    enc = tok(s, return_tensors="pt", truncation=True, max_length=320).to(device)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_num_classes=len(IDS))
    lg = out.logits.flatten()
    return {l: float(lg[j].item()) for j, l in enumerate(IDS)}


def micro(y_true, y_pred):
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        t, p = set(t), set(p)
        tp += len(t & p); fp += len(p - t); fn += len(t - p)
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    return {"micro_f1": round(2*pr*rc/max(1e-9, pr+rc), 4), "precision": round(pr, 4), "recall": round(rc, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=str(MODEL_DIR))
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--demo", nargs="*", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiClassModel.from_pretrained(args.model_dir).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.model_dir, add_prefix_space=True)

    if args.demo is not None:
        for c in args.demo:
            sc = score_all(model, tok, c, device)
            verdict = max(VERDICT, key=lambda l: sc.get(l, -99))
            acts = [a for a in ACTIONS if sc.get(a, -99) >= args.threshold]
            att = schema_v2.derive_attck(acts)
            print(f"\n{c}\n  verdict  : {verdict}\n  actions  : {acts}\n  tactics  : {att['tactics']}\n  techniques: {att['techniques']}")
        return

    data = json.loads((Path(DATA_DIR) / f"{args.split}.json").read_text())
    if args.limit:
        data = data[: args.limit]
    print(f"evaluating {len(data)} {args.split} samples on {device} ...", flush=True)

    yt_act, yp_act, v_ok = [], [], 0
    for k, ex in enumerate(data):
        sc = score_all(model, tok, ex["text"], device)
        true = set(ex["true_labels"])
        v_true = [v for v in VERDICT if v in true]
        v_pred = max(VERDICT, key=lambda l: sc.get(l, -99))
        if v_true and v_true[0] == v_pred:
            v_ok += 1
        yt_act.append([a for a in ACTIONS if a in true])
        yp_act.append([a for a in ACTIONS if sc.get(a, -99) >= args.threshold])
        if (k + 1) % 1000 == 0:
            print(f"  {k+1}/{len(data)}", flush=True)

    print("verdict_acc :", round(v_ok / max(1, len(data)), 4))
    print("action       :", micro(yt_act, yp_act))


if __name__ == "__main__":
    main()
