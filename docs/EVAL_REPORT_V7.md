# HyperIDs v7（v1.5.1）评测报告

> 报告日期：2026-10-05
> 评测对象：`model/checkpoints_gpu/final_model_v7`（= GitHub release v1.5.1）
> 模型：GLiClass + `prajjwal1/bert-small`（17.6M，≤30M），512 长度

## 1. 本轮改动（相对 v6 / v1.5.0）

| 改动 | 说明 |
|---|---|
| hard-negative 回灌 | 用 gpt-5.6-sol 仲裁红队歧义样本，回灌「gt=malicious & gpt=malicious」双确认 **446 条** |
| verdict 强制 malicious | 回灌样本 verdict=malicious，actions 用 Jev actions |
| final_model_v7 | 从 v6 继续训练 4 epochs（L4 GPU，batch 64，lr 5e-5），test 冻结 6,287 |

## 2. 内测指标（action 平衡 test 6,287，v6 vs v7 同一 test 集）

| 指标 | v6 | v7 | Δ |
|---|---:|---:|---:|
| verdict_acc | 0.9302 | 0.9278 | -0.0024 |
| action micro-Precision | 0.9105 | 0.9109 | +0.0004 |
| action micro-Recall | 0.9206 | 0.9157 | -0.0049 |
| action micro-F1 | 0.9155 | 0.9133 | -0.0022 |
| action macro-F1 | 0.8280 | **0.8331** | **+0.0051** |

## 3. 弱类变化（重点）

| action | v6 F1 | v7 F1 | n(test) |
|---|---:|---:|---:|
| action.self_propagate | 0.571 | **0.629** | 20 |
| action.ransomware | 0.500 | **0.552** | 20 |
| action.web_shell | 0.571 | 0.558 | 110 |
| action.keylog | 0.526 | **0.600** | 52 |

其他改善：process_inject 0.871→0.900、brute_force 0.863→0.880、clear_logs 0.907→0.944。
回退项：bind_shell 0.798→0.757、account_add 0.935→0.901、credential_dump 0.902→0.889。

## 4. 诚实结论

1. **macro-F1 0.8280 → 0.8331（+0.51），弱类 3 个提升**：self_propagate / ransomware / keylog 都涨，
   方向正确；web_shell 基本持平。
2. **micro-F1 与 verdict_acc 小幅回退**（-0.22 / -0.24）：回灌把部分「Jev 判轻但 gt/gpt 判恶意」
   的样本强制成 malicious，模型对少数边界样本变激进，precision 波动；bind_shell / account_add 有回退。
3. 整体是**边际正收益**：macro 创历史新高，但幅度小于 v5→v6；hard-negative 回灌对 verdict 置信度
   有帮助，但 446 条相对 66k train 仍偏少，且 Jev action 标签在这些样本上有噪声。
4. 下一步更值得做的是：扩大回灌规模（更多 gpt 确认恶意的样本）+ 对 ransomware/keylog/web_shell
   专项合成，而不是继续同规模回灌。

## 5. 部署产物（v1.5.1）

| 项 | 值 |
|---|---|
| INT8 权重 | `model_int8.bin` 17.6MB（per-channel int8，2D 量化 / 1D fp32） |
| 静态单二进制 | `build/hyperids` ~18.4MB（statically linked） |
| 峰值 RSS | ~38 MiB（<100MB 目标） |
| 推理速度 | ~280 ms/条（AVX2+FMA，CPU，x86-64-v3） |

## 6. 复现

```bash
# 回灌（本地已生成 gliclass_v4，test 冻结）
PYTHONPATH=. python -m hyperids.data.merge_hardneg --prepared adversarial/runs/20260930-v5-hardneg-audit/hardneg_446.jsonl

# 训练（GPU，L4）
python -m hyperids.train --data_dir dataset/gliclass_v4 \
  --save_name final_model_v7 --resume_from model/checkpoints/final_model_v6 \
  --epochs 4 --batch_size 64 --lr 5e-5 --device cuda

# 评估
python -m hyperids.eval --model_dir model/checkpoints_gpu/final_model_v7 --split test
```
