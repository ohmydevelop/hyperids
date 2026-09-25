# Frontier LLM 模型选型结论

网关：`https://ai-api-gateway.app.baizhi.cloud/api/openai/chat/completions`（同时支持 `/responses`）
Key：`.key`（已确认可用，39 个模型）
原始数据：`configs/model_bench.json` / `configs/model_bench_v2.json`

## 选型结论（两档）

| 角色 | 模型 | 依据 |
|---|---|---|
| **主力标注/生成（默认）** | `gpt-5.6-sol` | v1: risk 7/8 + tech 3/4 + intent 4/4，~5.3s，0 error；v2 综合分最高，支持 `response_format=json_object` |
| **快速批量合成** | `qwen-flash` | ~0.9–4s，最稳定便宜，适合 obfuscation / minimal-pair 等大批量变体 |
| **高难标注/红队（备选）** | `deepseek-v4-pro` | 推理能力强、v1 满分，但 ~16s 且带 reasoning token，按需用 |

## 关键发现（直接影响 pipeline 设计）

1. **全量 199 标签塞进 prompt 会显著拉低准确率**（v2 与 v1 对比）。
   → pipeline 必须**分标签组提示 / 候选标签检索**，不能每次全量。
2. **模型输出不是稳定 canonical id**（可能 `"malicious"`、`"suspicious"`、自由文本）。
   → 必须有 **normalize + fuzzy 映射 + schema 校验** 层，非法输出回退重试。
3. 网关在 24 并发下会出现 502；`gpt-5.4-mini` 单独测也 502。
   → 客户端并发控制在 **≤8**，并对 5xx 退避重试。
4. `response_format={"type":"json_object"}` 可用，能提升 JSON 稳定性。

## 客户端约定

- `temperature=0`（生成多样变体时单独提温）
- `response_format={"type":"json_object"}`
- 超时 30s，5xx 指数退避重试 3 次
- 并发 ≤8

---

## Jev（Teacher / 标注引擎）补充结论

Key 已保存到 `.jev_key`（chmod 600），端点 `https://api.typesafe.ai/v1/systemone`，模型 `jev-latest` → `jev-1.13.0`。

### 接入方式
- Jev 不走 chat/completions，走 decisions 端点（`state + questions`），见 `data_pipeline/jev_client.py`。
- 三种题型：`choice`（≤255 选项，返回每选项概率 + confidence）、`noul`（是非概率）、`score`（≤10 级）。
- 并行评估：**197 个问题一次请求，耗时与 1 个问题相当（~1.2s）**。

### 199-label schema → Jev 映射
```text
risk      -> 1 个 Choice（3 选项，概率即 softmax）
intent    -> 41 个 Noul
tactic    -> 14 个 Noul
technique -> 141 个 Noul
共 197 题 → 199 维概率向量（schema.all_label_ids() 顺序）
```
见 `data_pipeline/jev_labels.py`。

### 8 命令套件实测（configs/jev_vs_gpt.json）
| 指标 | Jev (jev-1.13.0) | gpt-5.6-sol |
|---|---|---|
| risk | 4/8 | 6/8 |
| technique | 3/4 | 3/4 |
| intent | 4/4 | 4/4 |
| tactic | 4/4 | 4/4 |
| 平均延迟 | ~1.6s | ~4.7s |

40 条种子实测：Jev 10.5s 全部标注完（gpt ~37s），两者 risk 一致率 33/40。
Jev 对双用途命令更保守（多判 suspicious），风险边界与 gpt 基本一致。

### 结论
- **Teacher 软标签：用 Jev**（199 维概率向量直接喂 KD，快 ~3-4x、便宜 ~400x、且校准概率）。
- **Stage 2 标注：用 Jev** 替代 gpt-5.6-sol（吞吐质变）。
- **数据生成/混淆/hard-neg/红队：仍用 LLM**（Jev 只下判断、不生成文本）。
- 风险边界（benign/suspicious/malicious）是主观边界，用 soft 概率携带不确定性即可，不必强求与种子先验一致。
