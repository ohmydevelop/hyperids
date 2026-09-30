"""红队集 ground_truth 口径分析：malicious Recall + benign FPR。

依赖：raw/*.jsonl 已含 ground_truth（Jev 打标）+ benign_negative.jsonl。
用 predict.py（滑动窗口 + 512）对 benign 负样本推理，恶意样本复用
runs/*/results/all_annotated.jsonl 的 model_verdict。
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "adversarial" / "raw"
RUNS = ROOT / "adversarial" / "runs"


def load_annotated():
    """从最新的 run 读 model_verdict（按 command sha256）。"""
    runs = sorted(RUNS.glob("*/results/all_annotated.jsonl"))
    if not runs:
        return {}
    latest = runs[-1]
    out = {}
    for line in latest.read_text().splitlines():
        try:
            d = json.loads(line)
            out[d.get("_command_sha256", d.get("command",""))] = d.get("model_verdict")
        except Exception:
            pass
    print(f"复用 run {latest.parent.parent.name} 的 model_verdict（{len(out)} 条）")
    return out


def main():
    import hashlib
    annotated = load_annotated()

    gt_by_verdict = Counter()
    matched = 0
    # 恶意样本：ground_truth vs model_verdict
    for f in sorted(RAW.glob("*.jsonl")):
        if f.name == "benign_negative.jsonl":
            continue
        for line in f.read_text().splitlines():
            try: d = json.loads(line)
            except: continue
            gt = d.get("ground_truth")
            if gt not in ("malicious", "suspicious", "benign"):
                continue
            h = hashlib.sha256(d["command"].encode()).hexdigest()
            mv = annotated.get(h)
            if mv is None:
                continue
            matched += 1
            gt_by_verdict[(gt, mv.replace("verdict.", ""))] += 1

    # 恶意样本 Recall
    mal_total = sum(v for (gt,mv),v in gt_by_verdict.items() if gt=="malicious")
    mal_detected = sum(v for (gt,mv),v in gt_by_verdict.items() if gt=="malicious" and mv=="malicious")
    mal_loose = sum(v for (gt,mv),v in gt_by_verdict.items() if gt=="malicious" and mv in ("malicious","suspicious"))
    sus_total = sum(v for (gt,mv),v in gt_by_verdict.items() if gt=="suspicious")
    print(f"\n恶意样本（ground_truth 匹配 {matched} 条）:")
    print(f"  malicious 严格 Recall: {mal_detected/max(1,mal_total)*100:.1f}%  ({mal_detected}/{mal_total})")
    print(f"  malicious 宽松 Recall: {mal_loose/max(1,mal_total)*100:.1f}%")
    if sus_total:
        sus_det = sum(v for (gt,mv),v in gt_by_verdict.items() if gt=="suspicious" and mv in ("malicious","suspicious"))
        print(f"  suspicious 召回(非benign): {sus_det/sus_total*100:.1f}%  ({sus_det}/{sus_total})")

    # benign 负样本：需要单独评测
    print(f"\nbenign 负样本需评测（见 evaluate_benign.py）")


if __name__ == "__main__":
    main()
