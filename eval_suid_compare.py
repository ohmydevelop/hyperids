"""Compare a SUID-augmented model against the baseline on:
  1. in-domain test (verdict_acc + action micro-F1)
  2. external GTFOBins shell_escape subset (malicious / nonbenign recall)

Usage:
  python eval_suid_compare.py --model_dir model/checkpoints_gpu/final_model_v2_suid \
      --baseline_dir model/checkpoints_gpu/final_model_v2
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from gliclass import GLiClassModel
from transformers import AutoTokenizer
import schema

ROOT = Path(__file__).resolve().parent
IDS = schema.all_label_ids()
OFF = schema.group_offsets()
VERDICT = IDS[OFF["verdict"][0]: OFF["verdict"][1]]
ACTIONS = IDS[OFF["action"][0]: OFF["action"][1]]


def load(model_dir, device):
    model = GLiClassModel.from_pretrained(str(model_dir)).to(device).eval()
    tok = AutoTokenizer.from_pretrained(str(model_dir), add_prefix_space=True)
    return model, tok


def predict_all(model, tok, cmds, device, bs=128):
    prefix = "".join(f"<<LABEL>>{l}" for l in IDS) + "<<SEP>>"
    out = []
    with torch.no_grad():
        for i in range(0, len(cmds), bs):
            ss = [prefix + c for c in cmds[i:i + bs]]
            enc = tok(ss, return_tensors="pt", truncation=True, max_length=320, padding=True)
            lg = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device),
                       max_num_classes=len(IDS)).logits.cpu()
            out.append(lg)
    logits = torch.cat(out, dim=0)
    res = []
    for row in logits:
        v_logits = row[:3]
        vp = torch.softmax(v_logits, dim=0)
        verdict = VERDICT[int(torch.argmax(vp))]
        acts = [a for j, a in enumerate(ACTIONS) if torch.sigmoid(row[3 + j]) >= 0.5]
        res.append({"verdict": verdict, "actions": acts})
    return res


def shell_escape(t):
    tl = t.lower()
    if re.search(r'attacker\.com|/dev/tcp|/dev/udp|nc\s+-e|busybox\s+nc|exec\s+bash\s+-i|>&\s*/dev/tcp|curl\s+.*attacker|wget\s+.*attacker', t, re.I):
        return "definite_payload"
    if re.search(r'curl|wget', t, re.I) and re.search(r'\|\s*(sh|bash)|;\s*(sh|bash)|-o\s+/tmp|--output', t, re.I):
        return "definite_payload"
    if re.search(r'system\(|exec\s+/bin/sh|/bin/sh\s*</dev|/bin/sh\s*1>&|!sh\b|!/bin/sh|-e\s*[\'\"]system|rsh-command', t, re.I):
        return "shell_escape"
    return "dual_use"


def eval_model(model_dir, device):
    model, tok = load(model_dir, device)

    # 1) in-domain test
    test = json.loads((ROOT / "dataset" / "gliclass_v2" / "test.json").read_text())
    preds = predict_all(model, tok, [e["text"] for e in test], device)
    v_ok = 0
    yt, yp = [], []
    for ex, p in zip(test, preds):
        tv = [v for v in VERDICT if v in ex["true_labels"]]
        if tv and tv[0] == p["verdict"]:
            v_ok += 1
        yt.append([a for a in ACTIONS if a in ex["true_labels"]])
        yp.append(p["actions"])
    tp = fp = fn = 0
    for t, p_ in zip(yt, yp):
        t, p_ = set(t), set(p_)
        tp += len(t & p_); fp += len(p_ - t); fn += len(t - p_)
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    f1 = 2 * pr * rc / max(1e-9, pr + rc)

    # 2) external gtfobins shell_escape
    gt = [json.loads(l) for l in (ROOT / "dataset" / "external_eval" / "gtfobins.jsonl").read_text().splitlines() if l.strip()]
    se = [r for r in gt if shell_escape(r["text"]) == "shell_escape"]
    se_preds = predict_all(model, tok, [r["text"] for r in se], device)
    mal = sum(1 for p in se_preds if p["verdict"] == "verdict.malicious")
    nonben = sum(1 for p in se_preds if p["verdict"] != "verdict.benign")

    return {
        "verdict_acc": round(v_ok / len(test), 4),
        "action_f1": round(f1, 4),
        "shell_escape_n": len(se),
        "shell_escape_malicious_recall": round(mal / len(se), 4),
        "shell_escape_nonbenign_recall": round(nonben / len(se), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--baseline_dir", type=str, default=str(ROOT / "model" / "checkpoints_gpu" / "final_model_v2"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    for tag, d in [("baseline", args.baseline_dir), ("suid", args.model_dir)]:
        p = Path(d)
        if not p.exists():
            print(f"{tag}: {d} NOT FOUND, skip")
            continue
        r = eval_model(p, device)
        print(f"\n[{tag}] {d}")
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
