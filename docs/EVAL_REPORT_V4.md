# HyperIDs v4（v1.3.0）评测报告

> 报告日期：2026-09-30
> 评测对象：`model/checkpoints_gpu/final_model_v4`（= GitHub release v1.3.0）
> 模型：GLiClass + bert-small，词表裁剪后 17.6M，512 长度，verdict 损失加权

## 1. 模型改动（相对 v3 / v1.2.0）

| 改动 | 说明 |
|---|---|
| 词表裁剪 | 30,522 → 6,702 token，参数量 29.8M → 17.6M |
| 512 长度 | 命令可用空间 125 → 317 token，截断率 10.9% → 0.5% |
| verdict 损失加权 | malicious 正样本 1.3x 权重 |
| 滑动窗口分段 | 超长命令 120-token 窗口、步长 90，verdict 取最恶意段、action 取 max |

## 2. 内测指标（test 5,978）

| 指标 | v1.2.0 | v1.3.0 |
|---|---|---|
| verdict_acc | 0.9244 | 0.9244 |
| action micro-F1 | 0.9205 | 0.9214 |
| action precision | 0.9394 | 0.9422 |
| action recall | 0.9023 | 0.9014 |
| 严格 malicious Recall | 94.75% | **95.86%** |
| 严格 benign FPR（mal 误报） | 0.044% | 0.044% |

## 3. 截断对抗修复（最致命问题的解决）

`r3_long_prefix.jsonl`（1,500 条长前缀截断模板，恶意 payload 推到窗口外）：

| 口径 | v3 | v4 |
|---|---|---|
| 严格 malicious 召回 | 0% | ~10% |
| 宽松（sus+mal）召回 | 0% | **~95%** |
| benign 漏检 | 100% | ~5% |

→ 320-token 截断导致的「恶意动作被截断后完全漏检」已基本解决。

## 4. 红队集 ground_truth 口径

红队集：3,620 恶意候选（Jev 打标 ground_truth：malicious 2,385 / suspicious 1,056 / benign 179）
+ 1,500 benign 负样本（`benign_negative.jsonl`）。

### 4.1 恶意候选混淆矩阵（gt → 模型）

| | 判 malicious | 判 suspicious | 判 benign |
|---|---:|---:|---:|
| 真 malicious (2385) | 747 | 1530 | 108 |
| 真 suspicious (1056) | 238 | 769 | 49 |
| 真 benign (179) | 19 | 156 | 4 |

### 4.2 关键口径

| 口径 | 结果 |
|---|---|
| 恶意严格 Recall（判 malicious） | 31.3% |
| 恶意宽松 Recall（判 non-benign） | **95.5%** |
| suspicious 召回（非 benign） | 95.4% |
| benign 严格误报（1500 负样本） | **0.00%** |
| benign 宽松误报（1500 负样本） | 68.9% |
| benign 宽松误报（179 内嵌 benign） | 97.8% |

## 5. 核心结论

1. **端侧恶意告警几乎零误报**：1,500 条良性命令 0 条被判 malicious，严格 FPR = 0.00%。
2. **红队攻击宽松召回 95.5%**：精心构造的对抗样本，95.5% 至少被判 suspicious，只有 4.5% 漏判 benign。
3. **严格召回与误报无法两全**：模型在对抗样本上「宁可判 suspicious、不敢判 malicious」——
   严格口径召回仅 31.3%，宽松口径良性误报 68.9%~97.8%。`suspicious` 中间态被严重过度使用。
4. **根因**：bert-small（17.6M）对混淆/长前缀/wrapper 变体的「恶意」置信度不足，这是模型容量 vs Jev
   教师（更强模型）的差距，不是单一 bug；调阈值无法同时兼顾严格召回和误报。

## 6. 待办

- [ ] 稀有 action（cryptomining / process_inject / self_propagate）补数据——受 LLM 网关护栏/502 阻塞
- [ ] session 级建模（组合命令语义）——P0，单条命令无法捕获
- [ ] suspicious/malicious 边界——严格召回 31.3% 偏低
- [ ] C 二进制 vs Python 分段召回差异（int8 量化误差，75% vs 98%）

## 7. 复现

```bash
# 内测
python -m hyperids.eval --model_dir model/checkpoints_gpu/final_model_v4 --split test

# 红队集（滑动窗口评测）
python adversarial/runs/20260930-v4-r130/evaluate_v4_sliding.py --model-dir model/checkpoints_gpu/final_model_v4

# benign 负样本 FPR
PYTHONPATH=. python /tmp/evaluate_benign_fpr.py

# ground_truth 分析
python adversarial/analyze_ground_truth.py
```
