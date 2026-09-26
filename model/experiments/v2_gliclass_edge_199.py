"""Fine-tune the latest official GLiClass V3 edge checkpoint for HyperIDs.

Base model: knowledgator/gliclass-edge-v3.0
  - backbone : jhu-clsp/ettin-encoder-32m (ModernBERT-style, 10L/384H)
  - params   : ~32.7M (slightly above the 30M 'shot-in-the-dark', chosen because
               it is the smallest *official* GLiClass V3 and keeps INT8 RSS < 100MB)
  - scorer   : MLP scorer (pretrained, much stronger than the old dot scorer)

We load the full pretrained GLiClass checkpoint (zero-shot knowledge preserved),
then only flip problem_type to multi-label and fine-tune end-to-end.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer
from gliclass import GLiClassModel
from gliclass.training import TrainingArguments, Trainer
from gliclass.data_processing import DataCollatorWithPadding, GLiClassDataset, AugmentationConfig

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "gliclass"
SAVE_DIR = ROOT / "model" / "checkpoints"

BASE_MODEL = "knowledgator/gliclass-edge-v3.0"
ARCH = "uni-encoder"
PROBLEM = "multi_label_classification"
MAX_LENGTH = 320
MAX_LABELS = 60


def load_json(p: Path):
    with open(p) as f:
        return json.load(f)


def build_model(device):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_prefix_space=True)
    model = GLiClassModel.from_pretrained(BASE_MODEL)
    # The edge checkpoint already contains <<LABEL>> / <<SEP>> special tokens and
    # correct class/text token indices. We only switch to multi-label head.
    model.config.problem_type = PROBLEM
    model.to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"model params: {n/1e6:.2f}M  (base={BASE_MODEL})")
    return model, tokenizer


def compute_metrics(p):
    predictions, labels = p
    labels = labels.reshape(-1)
    predictions = predictions.reshape(-1)
    preds = (predictions > 0.5).astype(np.int64)
    labels = np.where(labels > 0.5, 1, 0).astype(np.int64)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"micro_f1": float(f1), "micro_precision": float(prec), "micro_recall": float(rec)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--test_run", action="store_true")
    ap.add_argument("--resume_from", type=str, default=None)
    ap.add_argument("--save_name", type=str, default="final_model_edge")
    args = ap.parse_args()

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.resume_from:
        model = GLiClassModel.from_pretrained(args.resume_from).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.resume_from, add_prefix_space=True)
        model.config.problem_type = PROBLEM
        print(f"resumed from {args.resume_from}")
    else:
        model, tokenizer = build_model(device)

    train_data = load_json(DATA_DIR / "train.json")
    val_data = load_json(DATA_DIR / "val.json")
    test_data = load_json(DATA_DIR / "test.json")
    if args.test_run:
        train_data, val_data = train_data[:200], val_data[:64]
    print(f"data: train={len(train_data)} val={len(val_data)} test={len(test_data)}")

    label_to_desc = {d["label"]: d for d in load_json(DATA_DIR / "labels_desc.json")}

    aug = AugmentationConfig(enabled=True)
    no_aug = AugmentationConfig(enabled=False)
    train_ds = GLiClassDataset(train_data, tokenizer, aug, label_to_desc,
                               MAX_LENGTH, PROBLEM, ARCH, add_description=True,
                               prompt_first=True, labels_tokenizer=None, max_labels=MAX_LABELS)
    val_ds = GLiClassDataset(val_data, tokenizer, no_aug, label_to_desc,
                             MAX_LENGTH, PROBLEM, ARCH, add_description=True,
                             prompt_first=True, labels_tokenizer=None, max_labels=MAX_LABELS)

    collator = DataCollatorWithPadding(device=device)

    training_args = TrainingArguments(
        output_dir=str(SAVE_DIR),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        others_lr=args.lr,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=50,
        save_steps=2000,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        report_to="none",
        fp16=(device == "cuda"),
        dataloader_num_workers=8,
        use_cpu=(device == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    trainer.train()
    final_dir = SAVE_DIR / args.save_name
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"saved -> {final_dir}")

    test_ds = GLiClassDataset(test_data, tokenizer, no_aug, label_to_desc,
                              MAX_LENGTH, PROBLEM, ARCH, add_description=True,
                              prompt_first=True, labels_tokenizer=None, max_labels=MAX_LABELS)
    metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print("TEST metrics:", metrics)


if __name__ == "__main__":
    main()
