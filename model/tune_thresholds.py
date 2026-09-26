"""Per-group threshold tuning on the validation set.

Chunked inference over all 199 labels, then for each group (risk/intent/
tactic/technique) sweep decision thresholds to maximize micro-F1, and save
configs/thresholds.json for inference.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from gliclass import GLiClassModel
from transformers import AutoTokenizer
from schema import all_label_ids, group_offsets, collapsed_label_ids, collapsed_group_offsets

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model" / "checkpoints_gpu" / "final_model_edge"
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


def score_all(model, tok, cmd: str, device="cuda"):
    scores = {}
    for i in range(0, len(IDS), CHUNK):
        chunk = IDS[i:i + CHUNK]
        s = "".join(f"<<LABEL>>{l}" for l in chunk) + "<<SEP>>" + cmd
        enc = tok(s, return_tensors="pt", truncation=True, max_length=320).to(device)
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        max_num_classes=len(chunk))
        lg = out.logits.flatten()
        for j, l in enumerate(chunk):
            if j < lg.shape[0]:
                scores[l] = lg[j].item()
    return scores


def sweep(scores_true, threshs):
    best = {}
    for g, labels in GROUPS.items():
        pairs = [(scores.get(l, -99.0), int(l in true["true"])) for scores, true in scores_true for l in labels]
        y = [p[1] for p in pairs]
        x = [p[0] for p in pairs]
        best_g = (None, -1.0)
        for th in threshs:
            tp = fp = fn = 0
            for xi, yi in zip(x, y):
                pred = 1 if xi >= th else 0
                if pred == 1 and yi == 1:
                    tp += 1
                elif pred == 1 and yi == 0:
                    fp += 1
                elif pred == 0 and yi == 1:
                    fn += 1
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            if f1 > best_g[1]:
                best_g = (th, f1)
        best[g] = {"threshold": best_g[0], "f1": round(best_g[1], 4)}
    return best


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=str(MODEL_DIR))
    ap.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--collapsed", action="store_true")
    args = ap.parse_args()
    if args.collapsed:
        _use_collapsed()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiClassModel.from_pretrained(args.model_dir).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.model_dir, add_prefix_space=True)

    data = json.loads((Path(args.data_dir) / "val.json").read_text())[: args.limit]
    print(f"evaluating {len(data)} val samples on {device} ...")

    scores_true = []
    for k, ex in enumerate(data):
        sc = score_all(model, tok, ex["text"], device)
        scores_true.append((sc, {"true": set(ex["true_labels"])}))
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(data)}", flush=True)

    threshs = [round(-3 + 0.25 * i, 2) for i in range(25)]  # -3 .. 3
    best = sweep(scores_true, threshs)
    print("best thresholds per group:")
    print(json.dumps(best, indent=2, ensure_ascii=False))

    out = ROOT / "configs" / "thresholds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, indent=2, ensure_ascii=False))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
