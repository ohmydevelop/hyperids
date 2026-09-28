"""Export HyperIDs (GLiClass uni-encoder, bert-small) weights to flat C-layout binaries.

Two outputs:
  model_f32.bin — fp32 (reference)
  model_int8.bin — per-channel int8 for 2D weights + fp32 for 1D (bias/LayerNorm)

int8 layout per 2D matrix [N,K]: scale[N] (fp32) then W_int8[N*K] (int8).
1D arrays stay fp32. Matrices are emitted in the exact order `load_model_mem`
in hyperids.c reads them.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
SAFE = ROOT / "model" / "checkpoints_gpu" / "final_model_v3" / "model.safetensors"
OUT = ROOT / "deploy" / "c_runtime" / "ci_assets"
VOCAB_SRC = ROOT / "model" / "checkpoints_gpu" / "final_model_v3" / "vocab.txt"

H = 512
L = 4
VOCAB = 6702

# 2D matrices are int8-quantized per-channel (per output row); 1D stays fp32.
INT8_2D = True


def q(arr):
    """per-channel int8 quantize -> (W_int8 [N,K], scale [N])."""
    wf = arr.astype(np.float32)
    maxa = np.abs(wf).max(axis=1, keepdims=True)
    scale = np.maximum(maxa / 127.0, 1e-12).astype(np.float32).reshape(-1)
    wq = np.clip(np.round(wf / scale[:, None]), -127, 127).astype(np.int8)
    return wq, scale


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with safe_open(str(SAFE), framework="pt") as f:
        t = {k: f.get_tensor(k) for k in f.keys()}

    def arr(name):
        return np.ascontiguousarray(t[name].float())

    # ordered list of matrix names with (dim) -> 2D or 1D
    order = [
        ("model.encoder_model.embeddings.word_embeddings.weight", 2),
        ("model.encoder_model.embeddings.position_embeddings.weight", 2),
        ("model.encoder_model.embeddings.token_type_embeddings.weight", 2),
        ("model.encoder_model.embeddings.LayerNorm.weight", 1),
        ("model.encoder_model.embeddings.LayerNorm.bias", 1),
    ]
    for l in range(L):
        b = f"model.encoder_model.encoder.layer.{l}"
        order += [
            (f"{b}.attention.self.query.weight", 2), (f"{b}.attention.self.query.bias", 1),
            (f"{b}.attention.self.key.weight", 2), (f"{b}.attention.self.key.bias", 1),
            (f"{b}.attention.self.value.weight", 2), (f"{b}.attention.self.value.bias", 1),
            (f"{b}.attention.output.dense.weight", 2), (f"{b}.attention.output.dense.bias", 1),
            (f"{b}.attention.output.LayerNorm.weight", 1), (f"{b}.attention.output.LayerNorm.bias", 1),
            (f"{b}.intermediate.dense.weight", 2), (f"{b}.intermediate.dense.bias", 1),
            (f"{b}.output.dense.weight", 2), (f"{b}.output.dense.bias", 1),
            (f"{b}.output.LayerNorm.weight", 1), (f"{b}.output.LayerNorm.bias", 1),
        ]
    order += [
        ("model.text_projector.linear_1.weight", 2), ("model.text_projector.linear_1.bias", 1),
        ("model.text_projector.linear_2.weight", 2), ("model.text_projector.linear_2.bias", 1),
        ("model.classes_projector.linear_1.weight", 2), ("model.classes_projector.linear_1.bias", 1),
        ("model.classes_projector.linear_2.weight", 2), ("model.classes_projector.linear_2.bias", 1),
    ]

    f32_blobs = []
    i8_blobs = []
    for name, ndim in order:
        a = arr(name)
        f32_blobs.append(a.astype(np.float32).tobytes())
        if ndim == 2:
            wq, scale = q(a)
            i8_blobs.append(scale.tobytes())
            i8_blobs.append(wq.tobytes())
        else:
            i8_blobs.append(a.astype(np.float32).tobytes())

    f32_blob = b"".join(f32_blobs)
    i8_blob = b"".join(i8_blobs)
    (OUT / "model_f32.bin").write_bytes(f32_blob)
    (OUT / "model_int8.bin").write_bytes(i8_blob)

    cfg = {
        "vocab_size": VOCAB, "hidden_size": H, "num_layers": L, "num_heads": 8,
        "intermediate_size": 2048, "max_seq": 320, "layer_norm_eps": 1e-12,
        "num_labels": 31, "class_token_index": 30522, "text_token_index": 30523,
        "pad_token_id": 0, "hidden_act": "gelu",
    }
    (OUT / "model_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (OUT / "vocab.txt").write_text(VOCAB_SRC.read_text(encoding="utf-8"))

    meta = {
        "f32_bytes": len(f32_blob),
        "int8_bytes": len(i8_blob),
        "int8_sha256": hashlib.sha256(i8_blob).hexdigest(),
        "config": cfg,
    }
    (OUT / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote model_f32.bin ({len(f32_blob)/1e6:.1f}MB) + model_int8.bin ({len(i8_blob)/1e6:.1f}MB)")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
