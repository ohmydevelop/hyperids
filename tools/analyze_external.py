"""Post-hoc analysis of external-eval predictions.

The raw per-source verdict numbers are skewed by noisy extraction:
  - GTFOBins `functions[].code` is mostly *dual-use* file/encode/pack commands
    (single command text cannot be called malicious in isolation).
  - PayloadsAllTheThings markdown extraction pulls in prose lines.

This script splits the clearly-labeled malicious corpus into:
  definite_payload -> reverse/bind shell, download+exec, attacker C2
  shell_escape     -> SUID-privilege-escalation interpreter escapes
  dual_use         -> file read/encode/pack (ambiguous single-line)
and reports recall + ROC/PR-AUC against benign (NL2Bash) and dual-use.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

RES = Path("dataset/external_eval_results")


def load(name):
    return [json.loads(l) for l in (RES / f"{name}.predictions.jsonl").read_text().splitlines() if l.strip()]


def classify(t):
    tl = t.lower()
    if re.search(r'attacker\.com|/dev/tcp|/dev/udp|nc\s+-e|busybox\s+nc|exec\s+bash\s+-i|>&\s*/dev/tcp|curl\s+.*attacker|wget\s+.*attacker', t, re.I):
        return "definite_payload"
    if re.search(r'curl|wget', t, re.I) and re.search(r'\|\s*(sh|bash)|;\s*(sh|bash)|-o\s+/tmp|--output', t, re.I):
        return "definite_payload"
    if re.search(r'system\(|exec\s+/bin/sh|/bin/sh\s*</dev|/bin/sh\s*1>&|!sh\b|!/bin/sh|-e\s*[\'\"]system|rsh-command', t, re.I):
        return "shell_escape"
    return "dual_use"


def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return float("nan")
    o = np.argsort(s); y = y[o]
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = np.arange(1, len(y) + 1)[y]
    return float((ranks.sum() - npos * (npos + 1) / 2) / (npos * nneg))


def pr_auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    idx = np.argsort(-s); y = y[idx]
    npos = int(y.sum())
    if npos == 0:
        return float("nan")
    tp = np.cumsum(y); fp = np.cumsum(~y)
    prec = tp / np.maximum(1, tp + fp); rec = tp / npos
    rp = np.concatenate([[0.0], rec])
    return float(np.sum(prec * np.diff(rp)))


def main():
    gtf = load("gtfobins")
    nl = load("nl2bash")
    p = lambda r: r["verdict_probs"]["verdict.malicious"]

    definite = [r for r in gtf if classify(r["text"]) in ("definite_payload", "shell_escape")]
    dual = [r for r in gtf if classify(r["text"]) == "dual_use"]
    benign = nl

    print(f"definite_malicious={len(definite)}  dual_use={len(dual)}  benign={len(benign)}")

    for name, pos, neg in [("definite_malicious vs benign", definite, benign),
                           ("definite_malicious vs dual_use", definite, dual)]:
        y = [1] * len(pos) + [0] * len(neg)
        s = [p(r) for r in pos] + [p(r) for r in neg]
        print(f"{name}: ROC-AUC={auc(y,s):.4f}  PR-AUC={pr_auc(y,s):.4f}")

    for k in ("definite_payload", "shell_escape", "dual_use"):
        sub = [r for r in gtf if classify(r["text"]) == k]
        mal = sum(1 for r in sub if r["verdict"] == "verdict.malicious")
        non = sum(1 for r in sub if r["verdict"] in ("verdict.malicious", "verdict.suspicious"))
        avg = float(np.mean([p(r) for r in sub])) if sub else 0.0
        print(f"{k}: n={len(sub)} malicious_recall={mal/max(1,len(sub)):.3f} "
              f"nonbenign_recall={non/max(1,len(sub)):.3f} avg_P(mal)={avg:.3f}")


if __name__ == "__main__":
    main()
