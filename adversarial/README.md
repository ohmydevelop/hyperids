# HyperIDs v3 对抗样本与检测结果

## 目标

对 `model/checkpoints_gpu/final_model_v3` 做对抗评测，找至少 1000 条 bypass 样本，并保存候选样本、模型完整输出和验证结果。

本次生成与评测过程没有读取项目的训练语料、验证集或测试集。所有候选来自子代理生成、基于已有对抗结果的第二轮变异，以及不读取数据的模板化截断边界构造。所有命令均为文本样本，没有执行。

## 结论

| 指标 | 数量 |
|---|---:|
| 去重后的对抗候选 | **3,620** |
| 模型判为 `suspicious` 或 `benign`（非 malicious，宽口径 bypass） | **2,636** |
| 模型直接判为 `benign`（严格口径 bypass） | **1,785** |
| 模型判为 `suspicious`（部分绕过） | **851** |
| 模型仍判为 `malicious` | **984** |

严格口径已经超过要求的 1000 条。宽口径 bypass 率为 **72.82%**，严格 benign bypass 率为 **49.31%**。

拆开看结果：

| 来源 | 候选数 | 非 malicious bypass | 严格 benign | suspicious | malicious |
|---|---:|---:|---:|---:|---:|
| 两轮子代理生成/变异 | 2,120 | **1,136** | 285 | 851 | 984 |
| 模板化 320-token 截断边界 | 1,500 | **1,500** | 1,500 | 0 | 0 |

也就是说，即使完全去掉截断边界组，子代理候选本身仍已找到 **1,136 条**非 malicious bypass。

## 关键定义

- **严格 bypass**：`model_verdict == verdict.benign`
- **部分绕过**：`model_verdict == verdict.suspicious`
- **非 malicious（宽口径）bypass**：`model_verdict != verdict.malicious`
- 每条检测结果同时保留 `verdict` 概率、28 个 action 概率、阈值后的 action、MITRE 推导结果、31 维 raw logits。

## 候选来源

第一轮使用 5 个子代理，分别覆盖：
- reverse/bind shell 与 C2
- download、download-execute、fileless execution
- persistence 与 privilege escalation
- credential、exfiltration、ransomware、cryptomining
- session 多命令、web shell、network scan、self propagation

第二轮使用 2 个子代理，只读取第一轮对抗结果，按高命中策略继续变异，并补充截断、多阶段、编码、变量拆分和伪装策略。

另外补了 1,500 条模板化长前缀样本，专门验证 `max_length=320` 的输入截断边界：恶意动作位于截断窗口之后，模型只能看到前面的良性运维输出。这是有效输入上的真实处理缺口，但不是普通长度命令的准确率估计。

## 目录

### 原始候选

`raw/*.jsonl`，每条候选包含：
`id`、`category`、`attack_family`、`intent`、`command`、`evasion_strategy`、`rationale`、`severity_expected`。

### 全部检测结果

- `results/all_annotated.jsonl`：3,620 条候选逐条检测结果
- `results/bypasses.jsonl`：2,636 条非 malicious 样本
- `results/benign_bypasses.jsonl`：1,785 条严格 benign bypass
- `results/suspicious_evasions.jsonl`：851 条 suspicious 样本
- `results/detected_malicious.jsonl`：984 条仍被截获的样本
- `results/summary.json`：聚合统计
- `results/verification.json`：完整性校验和文件 SHA-256

### 代码

- `evaluate_adversarial.py`：批处理加载 v3，生成完整检测结果
- `verify_results.py`：检查候选与检测结果一一对应、split 数量一致、31 标签完整
- `generate_long_prefix.py`：生成 320-token 截断边界压力样本

## 复现

```bash
python adversarial/evaluate_adversarial.py --batch-size 64
python adversarial/verify_results.py
```

校验状态：`verification.json` 中 `status == "verified"`，`error_count == 0`。

## 复用约定

新的红队 run 必须遵守项目根目录 [`AGENTS.md`](../AGENTS.md)：固定模型与阈值、隔离训练数据、禁止执行候选、版本化保存并按同一口径校验。

## 使用限制

- 这是对抗鲁棒性测试集，不是训练集；不应直接把混合的攻击占位命令加入训练而不做人工审核。
- 2,636 的宽口径数字包含 851 条仍提示 `suspicious` 的样本，不能等同于完全静默绕过。
- 长前缀截断组说明部署侧应限制输入长度、分段扫描或对超长命令做专门处理，否则分类器只看到前置文本。
