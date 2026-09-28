# 样本清单（Sample Manifest）

> 分类维度：`来源(source) × 标注(raw/labeled)`。
> 命名按数据源（quasarnix / nl2bash / synthetic / ...），不再是实现细节名。
> `corpus/` 为 git 同步的正式语料；`dataset/` 根目录为工作副本（gitignored）。

## 标注
- **raw**：原始样本，未打标签。
- **labeled**：Jev 打过标签（软/硬标签）。

## 来源
- **public**：`quasarnix` / `nl2bash` / `benign` / `suspicious` / `gtfobins_payloads`。
- **synthetic**：`synthetic` / `diverse` / `suid_variants` / `longtail`。

## 文件清单
- `corpus/labeled/public/benign.jsonl`  (23.16MB, 15897 条)
- `corpus/labeled/public/gtfobins_payloads.jsonl`  (0.24MB, 911 条)
- `corpus/labeled/public/nl2bash.jsonl`  (15.41MB, 10604 条)
- `corpus/labeled/public/quasarnix.jsonl`  (66.88MB, 32677 条)
- `corpus/labeled/public/suspicious.jsonl`  (13.98MB, 8820 条)
- `corpus/labeled/synthetic/diverse.jsonl`  (10.25MB, 6087 条)
- `corpus/labeled/synthetic/longtail.jsonl`  (1.27MB, 3826 条)
- `corpus/labeled/synthetic/suid_variants.jsonl`  (0.03MB, 277 条)
- `corpus/labeled/synthetic/synthetic.jsonl`  (3.23MB, 1806 条)
- `corpus/raw/public/benign.jsonl`  (2.03MB, 15897 条)
- `corpus/raw/public/gtfobins_payloads.jsonl`  (0.11MB, 915 条)
- `corpus/raw/public/nl2bash.jsonl`  (1.26MB, 10604 条)
- `corpus/raw/public/quasarnix.jsonl`  (7.63MB, 31534 条)
- `corpus/raw/public/suspicious.jsonl`  (1.12MB, 8820 条)
- `corpus/raw/synthetic/diverse.jsonl`  (0.82MB, 6087 条)
- `corpus/raw/synthetic/suid_variants.jsonl`  (0.05MB, 345 条)
- `corpus/raw/synthetic/synthetic.jsonl`  (0.87MB, 1806 条)

## 历史 parquet（早期 seed + 早期标签，已被 corpus 覆盖但保留溯源）

- `corpus/legacy/seed_labeled.parquet`（40 条 seed 原始）
- `corpus/legacy/seed_soft_jev.parquet`（40 条 seed Jev 软标签）
- `corpus/legacy/seed_hard_jev.parquet`（40 条 seed Jev 硬标签）
- `corpus/legacy/soft_labels.parquet`（226 条早期软标签）
- `corpus/legacy/synthetic.parquet`（18 条早期合成）
