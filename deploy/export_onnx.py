"""Export the final GLiClass model (bert-small, 31 labels) to ONNX.

The uni-encoder scores all 31 labels in one forward (CHUNK=31).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model" / "checkpoints_gpu" / "final_model_v9"
OUT_ONNX = MODEL_DIR / "model.onnx"

CHUNK = 31
SEQ_LEN = 256


class LogitsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=str(MODEL_DIR))
    ap.add_argument("--out", type=str, default=str(OUT_ONNX))
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--seq_len", type=int, default=SEQ_LEN)
    args = ap.parse_args()

    from gliclass import GLiClassModel
    from transformers import AutoTokenizer
    from hyperids.schema import all_label_ids

    model = GLiClassModel.from_pretrained(args.model_dir).eval()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    IDS = all_label_ids()

    # Match hyperids/predict.py exactly:
    #   [<<LABEL>>id ... <<SEP>>] + tokenized command + [SEP], padded to seq_len.
    # No [CLS] and no trailing [SEP] are added by the tokenizer.
    labels = IDS[: args.chunk]
    prefix = "".join(f"<<LABEL>>{l}" for l in labels) + "<<SEP>>"
    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    sample_ids = tok("sample command", add_special_tokens=False)["input_ids"]
    ids = (prefix_ids + sample_ids + [tok.sep_token_id])[: args.seq_len]
    pad_len = args.seq_len - len(ids)
    input_ids = torch.tensor([ids + [tok.pad_token_id] * pad_len], dtype=torch.long)
    attention_mask = torch.tensor([[1] * len(ids) + [0] * pad_len], dtype=torch.long)
    print("input shape:", tuple(input_ids.shape), "seq_len:", args.seq_len, "real_tokens:", len(ids))

    class AlignedWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask,
                              max_num_classes=args.chunk).logits

    wrapped = AlignedWrapper(model)
    with torch.no_grad():
        logits = wrapped(input_ids, attention_mask)
    print("logits shape (torch):", tuple(logits.shape))

    torch.onnx.export(
        wrapped,
        (input_ids, attention_mask),
        str(args.out),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}},
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
