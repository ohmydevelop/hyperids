# 样本清单（Sample Manifest）

> 分类维度：`来源(source) × 标注(raw/labeled)`。
> `corpus/` 为 git 同步的正式语料（31 标签：verdict 3 + action 28）。

## 标注
- **raw**：原始样本，未打标签。
- **labeled**：Jev 打过标签（verdict + actions + soft）。

## 来源
- **public**：`quasarnix` / `nl2bash` / `benign` / `suspicious` / `gtfobins` / `payloads` / `curated`。
- **synthetic**：`synthetic` / `diverse` / `suid_variants` / `longtail`。

## 文件清单
- `corpus/labeled/public/benign.jsonl`  (5.78MB, 15897 条)
- `corpus/labeled/public/curated.jsonl`  (0.01MB, 52 条)
- `corpus/labeled/public/gtfobins.jsonl`  (0.23MB, 811 条)
- `corpus/labeled/public/nl2bash.jsonl`  (3.71MB, 10604 条)
- `corpus/labeled/public/payloads.jsonl`  (0.01MB, 48 条)
- `corpus/labeled/public/quasarnix.jsonl`  (18.01MB, 32677 条)
- `corpus/labeled/public/suspicious.jsonl`  (3.37MB, 8820 条)
- `corpus/labeled/synthetic/diverse.jsonl`  (2.54MB, 6087 条)
- `corpus/labeled/synthetic/longtail.jsonl`  (1.27MB, 3826 条)
- `corpus/labeled/synthetic/suid_variants.jsonl`  (0.03MB, 277 条)
- `corpus/labeled/synthetic/synthetic.jsonl`  (0.84MB, 1806 条)
- `corpus/raw/public/benign.jsonl`  (2.03MB, 15897 条)
- `corpus/raw/public/curated.jsonl`  (0.01MB, 53 条)
- `corpus/raw/public/gtfobins.jsonl`  (0.10MB, 814 条)
- `corpus/raw/public/nl2bash.jsonl`  (1.26MB, 10604 条)
- `corpus/raw/public/payloads.jsonl`  (0.01MB, 48 条)
- `corpus/raw/public/quasarnix.jsonl`  (7.63MB, 31534 条)
- `corpus/raw/public/suspicious.jsonl`  (1.12MB, 8820 条)
- `corpus/raw/synthetic/diverse.jsonl`  (0.82MB, 6087 条)
- `corpus/raw/synthetic/suid_variants.jsonl`  (0.05MB, 345 条)
- `corpus/raw/synthetic/synthetic.jsonl`  (0.87MB, 1806 条)
