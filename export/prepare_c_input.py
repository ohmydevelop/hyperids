"""Tokenize a command + first chunk of labels into C-consumable int64 .bin inputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from schema import all_label_ids

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model" / "checkpoints_gpu" / "final_model_electra"
IDS = all_label_ids()
SEQ_LEN = 384
CHUNK = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", type=str)
    ap.add_argument("--out_dir", type=str, default=str(ROOT / "export"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    labels = IDS[:CHUNK]
    s = "".join(f"<<LABEL>>{l}" for l in labels) + "<<SEP>>" + args.command
    enc = tok(s, return_tensors="np", truncation=True, max_length=SEQ_LEN, padding="max_length")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    enc["input_ids"].astype(np.int64).tofile(out / "input_ids.bin")
    enc["attention_mask"].astype(np.int64).tofile(out / "attention_mask.bin")
    print(f"wrote {out/'input_ids.bin'} + {out/'attention_mask.bin'} (shape [1,{SEQ_LEN}])")


if __name__ == "__main__":
    main()
