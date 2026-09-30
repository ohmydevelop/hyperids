# HyperIDs v6（v1.5.0）评测报告

> 报告日期：2026-09-30
> 评测对象：`model/checkpoints_gpu/final_model_v6`（= GitHub release v1.5.0）
> 模型：GLiClass + `prajjwal1/bert-small`（17.6M，≤30M），512 长度

## 1. 本轮改动（相对 v5 / v1.4.0）

| 改动 | 说明 |
|---|---|
| 长尾补数据 | glm-5.3-flash few-shot 改写 + Jev 打标，新增 214 条高质量弱类样本 |
| 增量合并 | 90% train / 10% val，**test 冻结 6,287**（与 v5 同一份，可比） |
| final_model_v6 | 从 v5 继续训练 4 epochs（L4 GPU，batch 64，lr 5e-5） |

新增样本按 action 分布：self_propagate 59 / web_shell 75 / keylog 46 / ransomware 35（合成后经
Jev 0.7 阈值过滤，去重 1 条）。

## 2. 内测指标（action 平衡 test 6,287，v5 vs v6 同一 test 集）

| 指标 | v5 | v6 | Δ |
|---|---:|---:|---:|
| verdict_acc | 0.9310 | 0.9302 | -0.0008 |
| action micro-Precision | 0.9092 | 0.9105 | +0.0013 |
| action micro-Recall | 0.9140 | 0.9206 | +0.0066 |
| action micro-F1 | 0.9116 | **0.9155** | +0.0039 |
| action macro-F1 | 0.8052 | **0.8280** | **+0.0228** |

## 3. 弱类变化（重点）

| action | v5 F1 | v6 F1 | n(test) |
|---|---:|---:|---:|
| action.self_propagate | 0.308 | **0.571** | 20 |
| action.ransomware | 0.462 | **0.500** | 20 |
| action.web_shell | 0.593 | 0.571 | 110 |
| action.keylog | 0.500 | **0.526** | 52 |

其他明显改善：process_inject 0.793→0.871、brute_force 0.840→0.863、
bind_shell 0.769→0.798、credential_dump 0.888→0.902、account_add 0.891→0.935、
network_scan 0.832→0.842、disable_security 0.788→0.824。

## 4. 结论

1. **补数据方向有效**：self_propagate 从 0.308 → 0.571（+26.3 点），train support 79 → 131。
2. **macro-F1 0.8052 → 0.8280（+2.28 点）**，micro-F1 0.9116 → 0.9155。
3. **仍偏弱**：ransomware（0.500）、web_shell（0.571）、keylog（0.526）——本轮新增量还不够，
   且部分合成样本被 Jev 判定为该 action 概率 <0.7 而丢弃；下一步需更高质量的专项种子 + 困难负样本。
4. verdict_acc 微降 0.0008，属于噪声级别，可接受。

## 5. 部署产物（v1.5.0）

| 项 | 值 |
|---|---|
| INT8 权重 | `model_int8.bin` 17.6MB（per-channel int8，2D 量化 / 1D fp32） |
| 静态单二进制 | `build/hyperids` ~18.4MB（statically linked） |
| 峰值 RSS | ~38 MiB（<100MB 目标） |
| 推理速度 | ~280 ms/条（AVX2+FMA，CPU，x86-64-v3） |

## 6. 复现

```bash
# 合成（LLM few-shot 改写，只生成候选）
PYTHONPATH=. python -m hyperids.data.synthesize_v6_longtail --seeds_per_action 4 --variants_per_seed 8 --workers 6

# Jev 打标 + 过滤
PYTHONPATH=. python -m hyperids.data.label_v6_longtail --min_pos_prob 0.7 --workers 8

# 增量合并（test 冻结）
PYTHONPATH=. python -m hyperids.data.merge_v6_longtail

# 训练（GPU，L4）
python -m hyperids.train --data_dir dataset/gliclass_v3 \
  --save_name final_model_v6 --resume_from model/checkpoints/final_model_v5 \
  --epochs 4 --batch_size 64 --lr 5e-5 --device cuda

# 评估
python -m hyperids.eval --model_dir model/checkpoints_gpu/final_model_v6 --split test
```
