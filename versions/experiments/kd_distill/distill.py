"""Knowledge distillation — Student learns Jev's 199-dim output distribution.

Loss = hard-label CE + alpha * soft-target KL, on:
    risk (3)      : softmax — CE(hard argmax) + KL(softmax || Jev risk probs)
    intent/tactic/technique (196) : sigmoid — BCE(hard) + BCE(soft target)
Both the hard and soft targets come from Jev (dataset/soft_labels.parquet).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from hyperids.schema_legacy import all_label_ids
from student.model import TinyGLiClass
from student import tokenizer as tok_mod

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "soft_labels.parquet"
CKPT_DIR = ROOT / "student" / "checkpoints"

ALL_IDS = all_label_ids()
ID2IDX = {lid: i for i, lid in enumerate(ALL_IDS)}
RISK_IDS = [lid for lid in ALL_IDS if lid.startswith("risk.")]


def _row_to_tensors(row, tok) -> tuple[list[int], np.ndarray, np.ndarray]:
    ids = tok_mod.encode_batch([row["text"]], tok)[0]
    hard = np.zeros(len(ALL_IDS), dtype=np.float32)
    for l in row["labels"]:
        if l in ID2IDX:
            hard[ID2IDX[l]] = 1.0
    soft = np.array(row["soft"], dtype=np.float32)
    return ids, hard, soft


def load_data(tok, device="cpu", batch_size: int = 32):
    df = pd.read_parquet(DATA)
    ds = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        ids, hard, soft = [], [], []
        for _, r in sub.iterrows():
            a, h, s = _row_to_tensors(r, tok)
            ids.append(a); hard.append(h); soft.append(s)
        X = torch.tensor(ids, dtype=torch.long, device=device)
        H = torch.tensor(np.stack(hard), dtype=torch.float32, device=device)
        S = torch.tensor(np.stack(soft), dtype=torch.float32, device=device)
        ds[split] = torch.utils.data.TensorDataset(X, H, S)
    loaders = {s: torch.utils.data.DataLoader(ds[s], batch_size=batch_size, shuffle=(s == "train"))
               for s in ds}
    return loaders, ds


def loss_fn(logits: torch.Tensor, hard: torch.Tensor, soft: torch.Tensor, alpha: float) -> tuple[torch.Tensor, dict]:
    risk_logits = logits[:, :3]
    rest_logits = logits[:, 3:]
    hard_risk = hard[:, :3]
    soft_risk = soft[:, :3]
    hard_rest = hard[:, 3:]
    soft_rest = soft[:, 3:]

    hard_risk_idx = hard_risk.argmax(dim=1)
    l_risk_ce = F.cross_entropy(risk_logits, hard_risk_idx)
    l_risk_kl = F.kl_div(F.log_softmax(risk_logits, dim=-1), soft_risk, reduction="batchmean")
    l_rest_ce = F.binary_cross_entropy_with_logits(rest_logits, hard_rest)
    l_rest_kl = F.binary_cross_entropy_with_logits(rest_logits, soft_rest)

    loss = (l_risk_ce + l_rest_ce) + alpha * (l_risk_kl + l_rest_kl)
    metrics = {"ce": float((l_risk_ce + l_rest_ce).item()),
               "kd": float((l_risk_kl + l_rest_kl).item()),
               "total": float(loss.item())}
    return loss, metrics


@torch.no_grad()
def evaluate(model, loader, device="cpu", thr=0.5):
    model.eval()
    risk_correct = 0
    risk_total = 0
    tp = fp = fn = 0
    for X, H, S in loader:
        X = X.to(device)
        logits = model(X)
        probs = model.logits_to_probs(logits)
        # risk
        pred_risk = logits[:, :3].argmax(1)
        true_risk = H[:, :3].argmax(1)
        risk_correct += (pred_risk == true_risk).sum().item()
        risk_total += X.size(0)
        # multi-label micro F1 over non-risk labels (196) + risk one-hot
        pred_hard = torch.cat([F.one_hot(pred_risk, 3).float(), (probs[:, 3:] >= thr).float()], dim=1)
        tp += ((pred_hard == 1) & (H == 1)).sum().item()
        fp += ((pred_hard == 1) & (H == 0)).sum().item()
        fn += ((pred_hard == 0) & (H == 1)).sum().item()
    risk_acc = risk_correct / max(1, risk_total)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return risk_acc, f1


def train(
    epochs: int = 30,
    lr: float = 3e-4,
    batch_size: int = 32,
    alpha: float = 1.0,
    device: str = "cpu",
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # (re)train tokenizer on the current corpus
    tok = tok_mod.train()
    tok = tok_mod.load()

    model = TinyGLiClass(vocab_size=tok.get_vocab_size()).to(device)
    n_params = model.num_parameters()
    print(f"model params: {n_params/1e6:.2f}M  vocab={tok.get_vocab_size()}")

    loaders, ds = load_data(tok, device, batch_size)
    print(f"data: train={len(ds['train'])} val={len(ds['val'])} test={len(ds['test'])}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_f1 = 0.0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for X, H, S in loaders["train"]:
            X, H, S = X.to(device), H.to(device), S.to(device)
            opt.zero_grad()
            logits = model(X)
            loss, _ = loss_fn(logits, H, S, alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * X.size(0)
        sched.step()
        train_loss = tot / len(ds["train"])
        v_risk, v_f1 = evaluate(model, loaders["val"], device)
        print(f"ep {ep:2d}  loss={train_loss:.4f}  val_risk={v_risk:.3f}  val_f1={v_f1:.3f}  ({time.time()-t0:.0f}s)")
        if v_f1 > best_f1:
            best_f1 = v_f1
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "vocab_size": tok.get_vocab_size()},
                       CKPT_DIR / "student_best.pt")

    print(f"\nbest val f1={best_f1:.3f}")
    # final test
    ck = torch.load(CKPT_DIR / "student_best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    t_risk, t_f1 = evaluate(model, loaders["test"], device)
    print(f"TEST  risk_acc={t_risk:.3f}  micro_f1={t_f1:.3f}")
    print(f"checkpoint → {CKPT_DIR / 'student_best.pt'}")


if __name__ == "__main__":
    import sys
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    train(epochs=epochs)
