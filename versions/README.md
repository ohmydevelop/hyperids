# 历史版本归档（仅追溯，不参与当前链路）

> 这些是历史训练实验与被否决方案的**快照**。除 schema 引用已统一为
> `hyperids.schema_legacy` 外，其余 import/路径仍指向当时的模块名
> （如 `data_pipeline`、`student.model`），需配合旧目录结构才能直接运行。
> 当前可复现链路见 `docs/TRAINING.md`。

## 训练演化

| 版本 | 基座/骨干 | 标签 | 结果 |
|---|---|---|---|
| v1 | electra-small（13.75M） | 199（risk+intent+tactic+technique） | overall 0.658 |
| v2 | gliclass-edge-v3.0 / Ettin-32m | 199 | overall 0.784 |
| v3 | bert-small（29.8M） | 132（technique 折叠） | overall 0.791 |
| **v4（最终，见 hyperids/）** | **bert-small（29.8M）** | **31（verdict 3 + action 28）** | **verdict_acc 0.917 / action F1 0.909** |

> 演化主线：199 个难学的 MITRE 标签 → 去噪/折叠 → 31 个「verdict + 客观动作」，
> MITRE 改为规则推导（`hyperids.schema.derive_attck`）。

## 目录

- `v1_electra_199/` — electra-small + 199 标签训练
- `v2_gliclass_edge_199/` — gliclass-edge + 199 标签训练 + vocab 剪枝
- `v3_bert_132/` — bert-small + 132 标签训练/评估/推理
- `experiments/kd_distill/` — 软标签 KD 蒸馏（**已否决**：soft 目标让 recall 崩）
- `experiments/redteam/` — 红队闭环（199 标签时代，已废弃）
