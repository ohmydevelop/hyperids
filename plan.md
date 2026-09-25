# HyperIDs — 恶意命令/脚本多标签分类

## 架构定案（v4：最终模型 = 微调 GLiClass，参数量 ≤30M）

```text
Frontier LLM (造数据)  →  synthesize / obfuscate / hard_neg / min_pair / red-team
      ↓ 海量命令/脚本
Jev Teacher (打软标签)  →  199 维校准概率（risk + intent + tactic + technique）
      ↓  hard labels + 199 维 soft probabilities
GLiClass 微调（最终模型）  →  预训练多标签分类器，hard CE + soft KL
      ↓
INT8 量化 → 部署（RSS 后期优化）
```

### 职责边界

| 模型 | 角色 | 干什么 |
|---|---|---|
| Frontier LLM（gpt-5.6-sol / qwen-flash / feature-flash） | 教授 | 造样本 + 红队（不参与端侧推理） |
| Jev（TypeSafe `jev-1.13.0`） | Teacher | 199 维校准软标签（唯一软标签来源；LLM 标签仅作备份） |
| **GLiClass** | **最终模型** | 微调后 INT8 部署 |

### 硬约束（v4）

| 约束 | 值 |
|---|---|
| **参数量** | **≤30M** |
| RSS（运行时内存） | 后期优化 |
| INT8 体积 | 后期优化 |
| 标签空间 | 199（risk 3 + intent 41 + tactic 14 + technique 141） |

> 参数量硬约束 ≤30M（electra-small ~13.75M / bert-small ~29M）；RSS 与量化后期优化。
> 从零小模型（`student/`，4L/256D ~5M）保留为「极致端侧」备选。

---

## 标签空间（不变）

```yaml
risk: 3        # risk.benign / risk.suspicious / risk.malicious（互斥，softmax）
intent: 41     # 多标签
tactic: 14     # MITRE tactic（多标签）
technique: 141 # MITRE technique id（多标签）
# 总计 199
```

> **schema 是全链路 contract**：Frontier LLM / Jev / GLiClass 共用 `label_schema.yaml`。
> label 描述即 GLiClass 的 label representation 输入。

---

## 数据计划（补平衡）

当前 50K 恶意偏多、suspicious 极少，GLiClass 微调对类别分布敏感，因此补一批平衡数据：

| 步骤 | 内容 | 目标 |
|---|---|---|
| 1 | 续跑 Jev 软标签（跳过已标、重打 402 空行） | 50K 全部 199 维软标签 |
| 2 | LLM 定向生成 suspicious + benign | +10K suspicious +15K benign |
| 3 | Jev 软标签新增部分 | 总 ~75K |
| 4 | 分层 split（按 risk 分层） | train/val/test |

**目标风险分布 ≈ malicious 40% / suspicious 25% / benign 35%**（不再恶意一边倒）。

---

## 实施计划

### Phase 1 — 数据地基 ✅

- `label_schema.yaml`（199）+ `schema.py` / `schema_check.py` ✅
- LLM 选型 + Jev 接入 + 数据管线（llm / normalize / prompts / seeds / label / label_jev / build_dataset / bulk_50k）✅
- 候选集 50,031 条（quasarnix 恶意 31.5K + nl2bash benign 10.6K + LLM 7.9K）✅
- Jev 软标签 24.8K（其中 1,143 条 402 空标签待重打）🔄

### Phase 2 — 打标签（Jev 唯一软标签来源）

| 交付物 | 状态 |
|---|---|
| 续跑 `bulk_50k label`（重打空行 + 标剩余 ~26K） | ⬜ |
| 补 balanced 数据（+10K suspicious +15K benign）→ Jev 标注 | ⬜ |
| `dataset/soft_labels.parquet`（~75K，按 risk 分层 split） | ⬜ |

### Phase 3 — GLiClass 微调（最终模型）

| 交付物 | 说明 |
|---|---|
| `model/`（原 `teacher/` 改名） | `gliclass` 库加载 GLiClass（edge/small 优先，base 备选） |
| `model/finetune.py` | hard CE + Jev soft KL（Lightning L4 GPU） |
| `model/eval.py` | risk acc / micro-F1 / per-group 指标 |
| 选档决策 | 效果优先，暂不卡 RSS |

### Phase 4 — 导出 + 红队闭环

| 交付物 | 说明 |
|---|---|
| `export/quantize_int8.py` | INT8（后期优化） |
| `export/benchmark_rss.py` | RSS 实测（后期优化） |
| `redteam/loop.py` | Frontier LLM 造对抗样本 → Jev 重评分 → GLiClass 再微调 |

---

## 完整流水线（v4）

```text
真实命令语料（QuasarNix 恶意 + NL2Bash benign）+ LLM 合成/补平衡
              ↓
      Jev Teacher（199 维软标签，唯一软标签来源）
              ↓
         hard CE + soft KL
              ↓
      GLiClass 微调（最终模型）
              ↓
      hard example mining → Frontier LLM 红队 → Jev 重评分 → 再微调
              ↓
            INT8（RSS 后期优化）
              ↓
            部署
```

---

## 关键原则

| 原则 | 理由 |
|---|---|
| Schema 先行 | 一次定义，LLM / Jev / GLiClass 共用 |
| 标签先于微调 | 没有 soft labels，KD/微调无从谈起 |
| 造数据靠 LLM，软标签靠 Jev，模型用 GLiClass | 各用所长 |
| 参数量 ≤30M | electra-small（~14M）优先 |
| 数据平衡优先于堆量 | GLiClass 微调对类别分布敏感 |
| 红队最后但持续 | 闭环迭代 |

---

## 引用

- [GLiClass](https://github.com/knowledgator/gliclass) — 最终模型（预训练多标签分类器）
- [Jev / TypeSafe AI](https://docs.typesafe.ai) — Teacher（软标签）
- [QuasarNix](https://github.com/dtrizna/QuasarNix) / NL2Bash — 真实命令语料
- [GLiNER2.5](https://github.com/fastino-ai/GLiNER2) — span/relation 第二支路（暂不进主链路）
