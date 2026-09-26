"""INT8 dynamic quantization of the ONNX model via onnxruntime."""
from __future__ import annotations

from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

ROOT = Path(__file__).resolve().parents[1]
MODEL_ONNX = ROOT / "model" / "checkpoints_gpu" / "final_model_edge" / "model.onnx"
OUT_ONNX = ROOT / "model" / "checkpoints_gpu" / "final_model_edge" / "model_int8.onnx"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, default=str(MODEL_ONNX))
    ap.add_argument("--out", type=str, default=str(OUT_ONNX))
    args = ap.parse_args()

    quantize_dynamic(
        model_input=args.inp,
        model_output=args.out,
        weight_type=QuantType.QInt8,
        extra_options={"EnableSubgraph": True},
    )
    sz = Path(args.out).stat().st_size + Path(args.out + ".data").stat().st_size
    print(f"quantized ONNX → {args.out}  ({sz/1e6:.1f}MB incl. external data)")
    print("NOTE: embeddings (Gather) stay fp32; MatMul weights -> int8.")


if __name__ == "__main__":
    main()
