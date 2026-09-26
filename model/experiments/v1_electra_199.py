"""Fine-tune GLiClass for HyperIDs 199-label multi-label classification.

Model: GLiClass uni-encoder on microsoft/deberta-v3-small (~44M) → INT8 ~44MB,
chosen to satisfy the RSS < 100MB hard constraint. If you want more capacity,
switch ENCODER to 'microsoft/deberta-v3-base' and re-check RSS.

Training data (from model/prepare_data.py):
    dataset/gliclass/{train,val,test}.json   -> [{"text":..., "all_labels":[...]}]
    dataset/gliclass/labels_desc.json        -> [{"label":..., "description":...}]
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer, AutoConfig
from gliclass import GLiClassModel, GLiClassModelConfig
from gliclass.training import TrainingArguments, Trainer
from gliclass.data_processing import DataCollatorWithPadding, GLiClassDataset, AugmentationConfig

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "gliclass"
SAVE_DIR = ROOT / "model" / "checkpoints"

ENCODER = "google/electra-small-discriminator"   # ~14M — 参数 ≤30M 约束
ARCH = "uni-encoder"
PROBLEM = "multi_label_classification"
MAX_LENGTH = 320
MAX_LABELS = 60
RISK_IDS = ("risk.benign", "risk.suspicious", "risk.malicious")


def load_json(p: Path):
    with open(p) as f:
        return json.load(f)


def build_model(device):
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    encoder_config = AutoConfig.from_pretrained(ENCODER)

    config = GLiClassModelConfig(
        encoder_config=encoder_config,
        encoder_model=ENCODER,
        class_token_index=len(tokenizer),
        text_token_index=len(tokenizer) + 1,
        example_token_index=len(tokenizer) + 2,
        pooling_strategy="avg",
        class_token_pooling="first",
        scorer_type="simple",
        max_num_classes=50,
        use_lstm=False,
        normalize_features=False,
        extract_text_features=False,
        architecture_type=ARCH,
        prompt_first=True,
        squeeze_layers=False,
        layer_wise=False,
        encoder_layer_id=-1,
        dropout=0.3,
        shuffle_labels=True,
        use_segment_embeddings=False,
        focal_loss_alpha=-1,
        focal_loss_gamma=-1,
        focal_loss_reduction="none",
        contrastive_loss_coef=0.0,
    )
    model = GLiClassModel(config, from_pretrained=True)
    model.config.problem_type = PROBLEM

    new_words = ["<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>"]
    tokenizer.add_tokens(new_words, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"model params: {n/1e6:.2f}M  (encoder={ENCODER})")
    return model, tokenizer


def compute_metrics(p):
    predictions, labels = p
    if PROBLEM == "multi_label_classification":
        # micro F1 computed manually (no sklearn dependency)
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
    return {}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--test_run", action="store_true", help="train on 200 samples to verify pipeline")
    ap.add_argument("--resume_from", type=str, default=None, help="continue from a fine-tuned checkpoint dir")
    args = ap.parse_args()

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.resume_from:
        model = GLiClassModel.from_pretrained(args.resume_from).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.resume_from)
        print(f"resumed from {args.resume_from}")
    else:
        model, tokenizer = build_model(device)

    # load data
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
    final_dir = SAVE_DIR / "final_model"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"saved → {final_dir}")

    # evaluate on test
    test_ds = GLiClassDataset(test_data, tokenizer, no_aug, label_to_desc,
                              MAX_LENGTH, PROBLEM, ARCH, add_description=True,
                              prompt_first=True, labels_tokenizer=None, max_labels=MAX_LABELS)
    metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print("TEST metrics:", metrics)


if __name__ == "__main__":
    main()
