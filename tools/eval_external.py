"""Out-of-distribution evaluation on public external corpora.

Sources (dataset/external_eval/*.jsonl):
  malicious       -> gtfobins.jsonl, payloads.jsonl (clearly offensive)
  benign          -> nl2bash.jsonl (clearly legitimate)
  attack_session  -> cowrie.jsonl (real honeypot attacker commands; text may
                     be benign recon, so it is reported as a distribution, not
                     a hard binary ground truth)

Reports per-source verdict distribution + hard-label recalls/FPR on the clearly
labeled sources, plus a global ROC/PR-AUC using P(verdict=malicious) as score.
No training is performed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gliclass import GLiClassModel
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyperids import schema  # noqa: E402

MODEL_DIR = ROOT / "model" / "checkpoints_gpu" / "final_model_v2"
EVAL_DIR = ROOT / "dataset" / "external_eval"
OUT_DIR = ROOT / "dataset" / "external_eval_results"

IDS = schema.all_label_ids()
OFF = schema.group_offsets()
VERDICT = IDS[OFF["verdict"][0]: OFF["verdict"][1]]
ACTIONS = IDS[OFF["action"][0]: OFF["action"][1]]


def load_model(model_dir, device):
    model = GLiClassModel.from_pretrained(str(model_dir)).to(device).eval()
    tok = AutoTokenizer.from_pretrained(str(model_dir), add_prefix_space=True)
    return model, tok


def score_batch(model, tok, cmds, device, prefix, bs=64):
    """Return list of dicts {label: logit} for each command."""
    out_logits = []
    with torch.no_grad():
        for i in range(0, len(cmds), bs):
            chunk = cmds[i:i + bs]
            ss = [prefix + c for c in chunk]
            enc = tok(ss, return_tensors="pt", truncation=True, max_length=320, padding=True)
            logits = model(input_ids=enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device),
                           max_num_classes=len(IDS)).logits
            out_logits.append(logits.cpu())
    all_logits = torch.cat(out_logits, dim=0)  # [N, 31]
    results = []
    for row in all_logits:
        results.append({l: float(row[j].item()) for j, l in enumerate(IDS)})
    return results


def softmax(d):
    z = np.array([d[v] for v in VERDICT])
    e = np.exp(z - z.max())
    p = e / e.sum()
    return {v: float(p[i]) for i, v in enumerate(VERDICT)}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def analyze_source(name, rows, results, thr):
    """Compute per-source verdict distribution and label metrics."""
    n = len(rows)
    verdict_counts = {v: 0 for v in VERDICT}
    verdict_preds = []
    malicious_probs = []
    n_act = []
    n_attck = 0
    for r, sc in zip(rows, results):
        vp = softmax(sc)
        v = max(VERDICT, key=lambda x: vp[x])
        verdict_counts[v] += 1
        verdict_preds.append(v)
        malicious_probs.append(vp["verdict.malicious"])
        acts = [a for a in ACTIONS if sigmoid(sc[a]) >= thr]
        n_act.append(len(acts))
        if schema.derive_attck(acts)["tactics"]:
            n_attck += 1
    out = {
        "n": n,
        "verdict_dist": {v: round(verdict_counts[v] / n, 4) for v in VERDICT},
        "mean_actions_active": round(float(np.mean(n_act)), 3),
        "attck_rule_hit_rate": round(n_attck / max(1, n), 4),
    }
    if "label" in rows[0]:
        label = rows[0]["label"]
        if label == "malicious":
            out["malicious_recall"] = round(verdict_counts["verdict.malicious"] / n, 4)
            out["nonbenign_recall"] = round((verdict_counts["verdict.malicious"] + verdict_counts["verdict.suspicious"]) / n, 4)
        elif label == "benign":
            out["benign_accuracy"] = round(verdict_counts["verdict.benign"] / n, 4)
            out["malicious_fpr"] = round(verdict_counts["verdict.malicious"] / n, 4)
            out["suspicious_rate"] = round(verdict_counts["verdict.suspicious"] / n, 4)
    return out, verdict_preds, malicious_probs


def auc(y_true, scores):
    """ROC-AUC via rank; y_true in {0,1}."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s)
    y = y[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, len(y) + 1)[y]
    return float((ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_auc(y_true, scores):
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    idx = np.argsort(-s)
    y = y[idx]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    prec = tp / np.maximum(1, tp + fp)
    rec = tp / n_pos
    # integrate precision over recall
    rec_prev = np.concatenate([[0.0], rec])
    area = float(np.sum(prec * np.diff(rec_prev)))
    return area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=str(MODEL_DIR))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_per_source", type=int, default=0, help="cap samples per source (0 = all)")
    ap.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.model_dir, device)
    prefix = "".join(f"<<LABEL>>{l}" for l in IDS) + "<<SEP>>"

    files = sorted(EVAL_DIR.glob("*.jsonl"))
    if not files:
        print("no external eval data found; run data_pipeline/fetch_eval.py first")
        return

    all_y, all_s = [], []  # for global binary AUC (malicious=1, benign=0)
    per_source = {}

    for fp in files:
        name = fp.stem
        rows = [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
        if args.max_per_source and len(rows) > args.max_per_source:
            # deterministic sample
            rng = np.random.RandomState(0)
            idx = rng.choice(len(rows), args.max_per_source, replace=False)
            rows = [rows[i] for i in sorted(idx)]
        print(f"\n=== {name} ({len(rows)} samples) on {device} ===", flush=True)
        results = score_batch(model, tok, [r["text"] for r in rows], device, prefix, args.batch)
        summary, verdict_preds, mal_probs = analyze_source(name, rows, results, args.threshold)
        per_source[name] = summary
        print(json.dumps(summary, indent=2))

        # save per-sample predictions
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.out_dir) / f"{name}.predictions.jsonl", "w", encoding="utf-8") as f:
            for r, sc, vp in zip(rows, results, [softmax(x) for x in results]):
                f.write(json.dumps({"text": r["text"], "source": r["source"], "label": r.get("label"),
                                    "verdict": max(VERDICT, key=lambda x: vp[x]),
                                    "verdict_probs": vp}, ensure_ascii=False) + "\n")

        # accumulate global binary labels for clearly-labeled sources
        if rows and rows[0].get("label") == "malicious":
            all_y += [1] * len(rows)
            all_s += mal_probs
        elif rows and rows[0].get("label") == "benign":
            all_y += [0] * len(rows)
            all_s += mal_probs

    if all_y:
        roc = auc(all_y, all_s)
        pr = pr_auc(all_y, all_s)
        print("\n=== global binary (malicious vs benign, score=P(malicious)) ===")
        print(f"  n_malicious = {sum(all_y)}  n_benign = {len(all_y) - sum(all_y)}")
        print(f"  ROC-AUC = {roc:.4f}   PR-AUC = {pr:.4f}")

    # write summary json
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.out_dir) / "summary.json"
    summary_path.write_text(json.dumps(per_source, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsummary -> {summary_path}")


if __name__ == "__main__":
    main()
