# 数据准备链路（上游，可选重跑）

> 默认「从 corpus 起步」复现无需重跑本链路；仅当需要**从零合成/重新打标**时使用。

## 公开数据拉取

| 脚本 | 来源 | 输出 |
|---|---|---|
| `hyperids.data.fetch_quasarnix` | QuasarNix 恶意命令（**TODO: 地址**） | corpus/raw/public/quasarnix.jsonl |
| `hyperids.data.fetch_public` | GTFOBins / PayloadsAllTheThings | public_candidates.jsonl |
| `hyperids.data.fetch_eval` | Cowrie / NL2Bash / GTFOBins / Payloads | external_eval/*.jsonl |

## LLM 合成

| 脚本 | 作用 |
|---|---|
| `hyperids.data.bulk_50k` | 50k 候选批量合成（coverage + diverse），Jev 199 打标 |
| `hyperids.data.synthesize` | 按 seed 标签组合合成（199 标签时代） |
| `hyperids.data.synthesize_longtail` | 长尾 action 合成 + Jev 31 打标 |
| `hyperids.data.generate_suid_variants` | SUID 提权逃逸变体（确定性模板，命令+硬标签） |

## Jev 打标

- `hyperids.jev_labels_legacy`：199 标签 → Jev SystemOne 问题映射（唯一软标签来源）
- `hyperids.jev_labels`：31 标签（verdict 3 + action 28）
- 客户端 `hyperids.jev_client`；key 在 `.jev_key`（gitignore）

## 关键依赖

- LLM 网关客户端：`hyperids.llm`（key 在 `.key`）
- prompt 模板：`hyperids.prompts`
- 种子命令：`hyperids.seeds`
- 199 标签规范化：`hyperids.normalize`

## 数据格式约定

- **raw**：`{"text": ..., "source": <quasarnix|synthetic|diverse|...>, ...}`（未打标）
- **labeled**：`{"text": ..., "source": ..., "labels": [199维], "soft": [199维], "model": "jev-..."}`
- 分片约定见 `dataset/MANIFEST.md`（raw/labeled × public/synthetic）
