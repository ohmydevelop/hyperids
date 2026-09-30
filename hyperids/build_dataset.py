"""从 corpus/labeled（31 维 Jev 软标签）构建 gliclass_v2 训练集。

corpus 每行：{text, source, verdict, actions, soft(31维), model}
输出 dataset/gliclass_v2/{train,val,test}.json + labels_desc.json，GLiClass 格式：
  {"text":..., "true_labels":[verdict]+actions, "all_labels":31, "soft_v2":31维}
"""
from __future__ import annotations

import json
import random
from collections import defaultdict, Counter
from pathlib import Path

from hyperids import schema

ROOT = Path(__file__).resolve().parents[1]
CORPUS_LABELED = ROOT / "dataset" / "corpus" / "labeled"
DST = ROOT / "dataset" / "gliclass_v2"

ALL_LABELS = list(schema.all_label_ids())
VERDICT = list(schema.verdict_ids())
ACTIONS = list(schema.action_ids())


def load_corpus(corpus_dir: Path = CORPUS_LABELED) -> list[dict]:
    rows = []
    seen = set()
    for p in sorted(corpus_dir.rglob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            text = d.get("text")
            verdict = d.get("verdict")
            if not text or verdict not in VERDICT or text in seen:
                continue
            seen.add(text)
            rows.append({
                "text": text,
                "verdict": verdict,
                "actions": [a for a in d.get("actions", []) if a in ACTIONS],
                "soft": d.get("soft"),
            })
    return rows


def stratified_split(rows: list[dict], val_ratio: float = 0.08, test_ratio: float = 0.08, seed: int = 42,
                     min_test_per_action: int = 20):
    """按 verdict 分层 split，再贪心补充保证每个 action 在 test 至少 min_test_per_action 条。"""
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["verdict"]].append(r)
    train, val, test = [], [], []
    rng = random.Random(seed)
    for v, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        test += items[:n_test]
        val += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]

    # 贪心补充：确保每个 action 在 test 至少 min_test_per_action 条
    test_actions = Counter(a for r in test for a in r["actions"])
    for action in ACTIONS:
        while test_actions[action] < min_test_per_action:
            moved = False
            for i, r in enumerate(train):
                if action in r["actions"]:
                    train.pop(i)
                    test.append(r)
                    test_actions[action] += 1
                    moved = True
                    break
            if not moved:
                break  # train 里没有含该 action 的样本了

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def build(corpus_dir: Path = CORPUS_LABELED, out_dir: Path = DST):
    rows = load_corpus(corpus_dir)
    print(f"valid corpus rows: {len(rows)}")
    train, val, test = stratified_split(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    desc = schema.descriptions()
    (out_dir / "labels_desc.json").write_text(
        json.dumps([{"label": l, "description": desc[l]} for l in ALL_LABELS],
                   ensure_ascii=False, indent=2))

    for name, split in (("train", train), ("val", val), ("test", test)):
        data = []
        for r in split:
            item = {
                "text": r["text"],
                "true_labels": [r["verdict"]] + r["actions"],
                "all_labels": ALL_LABELS,
            }
            if r["soft"] is not None and len(r["soft"]) == len(ALL_LABELS):
                item["soft_v2"] = r["soft"]
            data.append(item)
        (out_dir / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False))
        print(f"  {name}: {len(data)}")

    print(f"→ {out_dir}/ (train/val/test.json + labels_desc.json)")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", type=str, default=str(CORPUS_LABELED))
    ap.add_argument("--out", type=str, default=str(DST))
    args = ap.parse_args()
    build(Path(args.corpus_dir), Path(args.out))


if __name__ == "__main__":
    main()
