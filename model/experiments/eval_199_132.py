"""Grouped micro-F1 evaluation for the HyperIDs final model.

Chunked inference over all 199 labels, per-group decision thresholds from
configs/thresholds.json, risk exact-accuracy via argmax.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from gliclass import GLiClassModel
from transformers import AutoTokenizer
from schema_v1 import all_label_ids, group_offsets, collapsed_label_ids, collapsed_group_offsets

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model" / "checkpoints_gpu" / "final_model_edge"
THRESHOLDS = ROOT / "configs" / "thresholds.json"
DATA_DIR = ROOT / "dataset" / "gliclass"

IDS = all_label_ids()
OFF = group_offsets()
GROUPS = {g: IDS[OFF[g][0]: OFF[g][1]] for g in ("risk", "intent", "tactic", "technique")}

def _use_collapsed():
    global IDS, OFF, GROUPS
    IDS = collapsed_label_ids()
    OFF = collapsed_group_offsets()
    GROUPS = {g: IDS[OFF[g][0]: OFF[g][1]] for g in ("risk", "intent", "tactic", "technique")}
CHUNK = 20
SEQ_LEN = 320


def score_all(model, tok, cmd: str, device):
    scores = {}
    for i in range(0, len(IDS), CHUNK):
        chunk = IDS[i:i + CHUNK]
        s = "".join(f"<<LABEL>>{l}" for l in chunk) + "<<SEP>>" + cmd
        enc = tok(s, return_tensors="pt", truncation=True, max_length=SEQ_LEN).to(device)
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        max_num_classes=len(chunk))
        lg = out.logits.flatten()
        for j, l in enumerate(chunk):
            scores[l] = float(lg[j].item())
    return scores


def micro(y_true, y_pred):
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        t, p = set(t), set(p)
        tp += len(t & p)
        fp += len(p - t)
        fn += len(t - p)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"micro_f1": round(f1, 4), "micro_precision": round(prec, 4), "micro_recall": round(rec, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=str(MODEL_DIR))
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--thresholds", type=str, default=str(THRESHOLDS))
    ap.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--collapsed", action="store_true")
    args = ap.parse_args()
    if args.collapsed:
        _use_collapsed()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiClassModel.from_pretrained(args.model_dir).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.model_dir, add_prefix_space=True)
    th = json.loads(Path(args.thresholds).read_text())

    data = json.loads((Path(args.data_dir) / f"{args.split}.json").read_text())
    if args.limit:
        data = data[: args.limit]
    print(f"evaluating {len(data)} {args.split} samples on {device} ...", flush=True)

    y_true = {g: [] for g in GROUPS}
    y_pred = {g: [] for g in GROUPS}
    risk_exact = 0

    for k, ex in enumerate(data):
        scores = score_all(model, tok, ex["text"], device)
        true = set(ex["true_labels"])
        # risk exact (argmax among 3 mutually-exclusive risk labels)
        risk_true = [l for l in GROUPS["risk"] if l in true]
        risk_pred = max(GROUPS["risk"], key=lambda l: scores.get(l, -99.0))
        if risk_true and risk_true[0] == risk_pred:
            risk_exact += 1
        for g, labels in GROUPS.items():
            if g == "risk":
                pred = [l for l in labels if scores.get(l, -99.0) >= th[g]["threshold"]]
            else:
                pred = [l for l in labels if scores.get(l, -99.0) >= th[g]["threshold"]]
            y_true[g].append([l for l in labels if l in true])
            y_pred[g].append(pred)
        if (k + 1) % 1000 == 0:
            print(f"  {k+1}/{len(data)}", flush=True)

    out = {}
    all_t = [l for g in GROUPS for l in y_true[g]]
    all_p = [l for g in GROUPS for l in y_pred[g]]
    out["overall"] = micro(all_t, all_p)
    for g in GROUPS:
        out[g] = micro(y_true[g], y_pred[g])
    out["risk_acc"] = round(risk_exact / max(1, len(data)), 4)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
