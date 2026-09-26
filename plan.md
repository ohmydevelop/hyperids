# HyperIDs — 恶意命令/脚本分类（最终 v2）

## 架构定案（v6 / v2 schema —— 最终交付）

```text
Frontier LLM (造数据)  →  synthesize / obfuscate / hard_neg / red-team
      ↓ 海量命令/脚本
Jev Teacher (打软标签)  →  199 维校准概率（唯一软标签来源）
      ↓  确定性映射 199 -> 31
GLiClass 微调（最终模型）  →  prajjwal1/bert-small（29.8M），verdict 3 + action 28
      ↓
MITRE 规则表（derive_attck）  →  tactic / technique（确定性推导，不预测）
      ↓
INT8 量化 → ONNX Runtime C 部署（RSS < 100MB）
```

### 职责边界

| 模型/组件 | 角色 |
|---|---|
| Frontier LLM | 造样本 + 红队（不参与端侧推理） |
| Jev（jev-1.13.0） | 199 维校准软标签（唯一软标签来源） |
| **GLiClass + bert-small** | 最终模型：verdict（互斥）+ action（多标签） |
| **derive_attck 规则表** | action → tactic/technique 确定性映射 |

### 硬约束（最终，全部达成）

| 约束 | 值 |
|---|---|
| 参数量 | **29.8M（≤30M ✅）** |
| RSS | **INT8 C 运行时 96.2MB（<100MB ✅）** |
| INT8 体积 | **30.2MB** |
| 标签空间 | **31**（verdict 3 + action 28；MITRE 由规则推导，不算预测头） |

---

## 标签空间（v2 schema，最终）

```yaml
verdict: 3   # benign / suspicious / malicious（互斥，argmax）
action: 28   # 多标签：download, execute_local, download_execute,
             # command_and_control, obfuscate, file_operation, process_inject,
             # network_scan, brute_force, reverse_shell, bind_shell, web_shell,
             # backdoor, keylog, credential_dump, ransomware, cryptomining,
             # exfiltrate, disable_security, clear_logs, timestomp, account_add,
             # registry_persist, service_persist, schedule_persist, self_propagate,
             # system_probe, environment_setup
# MITRE tactic/technique：不预测，由 schema_v2.derive_attck(action) 规则表推导
```

> 设计原则：**决策互斥（verdict）+ 事实客观（action）+ MITRE 用规则推导**。
> 旧 199 标签（risk+intent+tactic+technique 141）存在 intent↔tactic 重复、子技术长尾、
> 标注不一致三个问题；v2 全部消除。

---

## 数据

- 来源：QuasarNix（恶意）+ NL2Bash（benign）+ LLM 合成 + 红队。
- Jev 199 维软标签：`dataset/soft_labels_50k.jsonl`（74,743 条）。
- `data_pipeline/build_dataset_v2.py`：199 软 → 31 软/硬（verdict 1:1，action 多对一求和）。
- 最终数据集 `dataset/gliclass_v2/`：train 65,480 / val 5,978 / test 5,978；28 action 全有样本。

---

## 最终结果（test 5,978）

| 指标 | fp32 | **INT8（部署）** |
|---|---|---|
| verdict_acc | 0.9167 | **0.9187** |
| action micro-F1 | 0.9090 | **0.9099** |
| action precision / recall | 0.935 / 0.884 | 0.934 / 0.887 |

- 对比旧体系 risk_acc：0.8694 → 0.917（+4.7pt）。
- INT8 与 fp32 基本无损（旧 132 标签 int8 会崩 risk；31 标签 + 简单 head 后消失）。

---

## 部署

- `model/checkpoints_gpu/final_model_v2/`：checkpoint + `model.onnx`（fp32 118MB）+ `model_int8.onnx`（30.2MB）。
- C 推理：`export/c_infer_example_v2.c`（N_LOGITS=31, SEQ_LEN=320）。
- RSS：fp32 114.7MB / **INT8 96.2MB**。
- 推理：31 标签一次前向打分 → verdict argmax + action 阈值 → `derive_attck` 出 MITRE。

---

## 关键文件

- `label_schema_v2.yaml` / `schema_v2.py` —— v2 schema + MITRE 规则表
- `data_pipeline/build_dataset_v2.py` / `build_dataset_v2_hard.py`
- `model/finetune_v2.py` / `model/eval_v2.py`
- `configs/`（阈值，如后续需要）、`export/README.md`
- 历史：`label_schema.yaml`（199）、`plan.md` 旧版、edge/electra 模型保留备查

## 后续可做（不在本次范围）

1. 软标签 KD（soft_v2 已生成，训练改用 dict 软标签再榨 action F1）。
2. 动作集按真实告警反馈继续打磨（26~28 个动作是否够、是否要加 `credential_dump` 之外的新动作）。
3. C 版 tokenizer + 单二进制发布。
