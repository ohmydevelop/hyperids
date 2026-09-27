"""Export HyperIDs (GLiClass uni-encoder, bert-small) weights to flat C-layout binary.

Layout (little-endian float32, C row-major, in this exact order):
  emb_words [30525,512], emb_pos [512,512], emb_ttype [2,512], emb_ln_w/b [512]
  4 x encoder layer: q,k,v,attn_o (w,b), attn_ln(w,b), ffn1(w,b), ffn2(w,b), ffn_ln(w,b)
  text_projector: linear_1(w,b), linear_2(w,b)
  classes_projector: linear_1(w,b), linear_2(w,b)
(bert pooler.dense and logit_scale are NOT used in inference.)
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
SAFE = ROOT / "model" / "checkpoints_gpu" / "final_model_v2" / "model.safetensors"
OUT = ROOT / "release" / "ci_assets"
VOCAB_SRC = ROOT / "model" / "checkpoints_gpu" / "final_model_v2" / "vocab.txt"

H = 512
L = 4
VOCAB = 30525


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with safe_open(str(SAFE), framework="pt") as f:
        t = {k: f.get_tensor(k) for k in f.keys()}

    def put(name):
        arr = np.ascontiguousarray(t[name], dtype=np.float32)
        return arr.tobytes()

    blobs = []
    blobs.append(put("model.encoder_model.embeddings.word_embeddings.weight"))
    blobs.append(put("model.encoder_model.embeddings.position_embeddings.weight"))
    blobs.append(put("model.encoder_model.embeddings.token_type_embeddings.weight"))
    blobs.append(put("model.encoder_model.embeddings.LayerNorm.weight"))
    blobs.append(put("model.encoder_model.embeddings.LayerNorm.bias"))
    for l in range(L):
        b = f"model.encoder_model.encoder.layer.{l}"
        blobs.append(put(f"{b}.attention.self.query.weight"))
        blobs.append(put(f"{b}.attention.self.query.bias"))
        blobs.append(put(f"{b}.attention.self.key.weight"))
        blobs.append(put(f"{b}.attention.self.key.bias"))
        blobs.append(put(f"{b}.attention.self.value.weight"))
        blobs.append(put(f"{b}.attention.self.value.bias"))
        blobs.append(put(f"{b}.attention.output.dense.weight"))
        blobs.append(put(f"{b}.attention.output.dense.bias"))
        blobs.append(put(f"{b}.attention.output.LayerNorm.weight"))
        blobs.append(put(f"{b}.attention.output.LayerNorm.bias"))
        blobs.append(put(f"{b}.intermediate.dense.weight"))
        blobs.append(put(f"{b}.intermediate.dense.bias"))
        blobs.append(put(f"{b}.output.dense.weight"))
        blobs.append(put(f"{b}.output.dense.bias"))
        blobs.append(put(f"{b}.output.LayerNorm.weight"))
        blobs.append(put(f"{b}.output.LayerNorm.bias"))
    # text_projector
    blobs.append(put("model.text_projector.linear_1.weight"))
    blobs.append(put("model.text_projector.linear_1.bias"))
    blobs.append(put("model.text_projector.linear_2.weight"))
    blobs.append(put("model.text_projector.linear_2.bias"))
    # classes_projector
    blobs.append(put("model.classes_projector.linear_1.weight"))
    blobs.append(put("model.classes_projector.linear_1.bias"))
    blobs.append(put("model.classes_projector.linear_2.weight"))
    blobs.append(put("model.classes_projector.linear_2.bias"))

    blob = b"".join(blobs)
    (OUT / "model_f32.bin").write_bytes(blob)

    cfg = {
        "vocab_size": VOCAB, "hidden_size": H, "num_layers": L, "num_heads": 8,
        "intermediate_size": 2048, "max_seq": 320, "layer_norm_eps": 1e-12,
        "num_labels": 31, "class_token_index": 30522, "text_token_index": 30523,
        "pad_token_id": 0, "hidden_act": "gelu",
    }
    (OUT / "model_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    # copy vocab
    (OUT / "vocab.txt").write_text(VOCAB_SRC.read_text(encoding="utf-8"))

    meta = {
        "binary_bytes": len(blob),
        "binary_sha256": hashlib.sha256(blob).hexdigest(),
        "config": cfg,
    }
    (OUT / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote model_f32.bin ({len(blob)/1e6:.1f}MB) + config + vocab -> {OUT}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
