"""INT8 quantization of the fine-tuned model (torch dynamic quant of Linear layers).

Saves a torch checkpoint with int8 weights and reports on-disk size. This is a
pragmatic INT8 pass; the authoritative memory number comes from benchmark_rss.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from gliclass import GLiClassModel

MODEL_DIR = ROOT / "model" / "checkpoints" / "final_model"
OUT_DIR = ROOT / "model" / "checkpoints" / "final_model_int8"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=str(MODEL_DIR))
    ap.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    model = GLiClassModel.from_pretrained(args.model_dir)
    # quantize all Linear layers to int8 (weights); embeddings stay fp32 unless specified
    q = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(q.state_dict(), out / "pytorch_model_int8.bin")
    size_mb = (out / "pytorch_model_int8.bin").stat().st_size / 1e6
    n = sum(p.numel() for p in q.parameters())
    print(f"params={n/1e6:.2f}M  int8 checkpoint={size_mb:.1f}MB  → {out}")
    print("NOTE: run `python3 -m export.benchmark_rss` for the authoritative RSS number.")


if __name__ == "__main__":
    main()
