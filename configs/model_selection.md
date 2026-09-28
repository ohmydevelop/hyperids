# Frontier LLM 模型选型结论

网关：`https://ai-api-gateway.app.baizhi.cloud/api/openai/chat/completions`
Key：`.key`（已确认可用，39 个模型）
原始数据：`configs/model_bench.json` / `configs/model_bench_v2.json`

## 选型结论

| 角色 | 模型 | 依据 |
|---|---|---|
| **主力生成/合成（默认）** | `gpt-5.6-sol` | 综合分最高，支持 `response_format=json_object`，稳定 |
| **快速批量合成** | `qwen-flash` | 最快最便宜，适合大批量变体 |
| **高难合成/红队（备选）** | `deepseek-v4-pro` | 推理能力强，但 ~16s 且带 reasoning token，按需用 |

## 关键发现

1. **过大的标签集塞进 prompt 会拉低准确率** → 打标 prompt 应精炼（当前 31 标签：1 choice + 28 noul）。
2. **模型输出不是稳定 canonical id** → 客户端需 schema 校验 + 非法输出重试。
3. 网关在 24 并发下会 502 → 客户端并发 ≤8，5xx 退避重试。
4. `response_format={"type":"json_object"}` 可用，能提升 JSON 稳定性。

## 客户端约定

- `temperature=0`（生成多样变体时单独提温）
- `response_format={"type":"json_object"}`
- 超时 30s，5xx 指数退避重试 3 次
- 并发 ≤8

---

## Jev（打标引擎）结论

Key 在 `.jev_key`（gitignore），端点 `https://api.typesafe.ai/v1/systemone`，模型 `jev-latest`。

- 走 decisions 端点（`state + questions`），见 `hyperids/jev_client.py`。
- 三种题型：`choice`（返回每选项概率）、`noul`（是非概率）、`score`（≤10 级）。
- **31 标签映射**：verdict → 1 个 Choice（3 选项），action → 28 个 Noul；共 29 题 → 31 维概率向量（`schema.all_label_ids()` 顺序），见 `hyperids/jev_labels.py`。
- Jev 对双用途命令更保守（多判 suspicious），风险边界用 soft 概率携带不确定性。
