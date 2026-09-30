# HyperIDs v3 评测交接报告

> 报告日期：2026-09-28  
> 评测对象：`model/checkpoints_gpu/final_model_v3`  
> 测试集：`dataset/gliclass_v2/test.json`  
> 结论：项目内测试集指标良好；生产泛化仍受 320-token 截断、组合命令语义和稀有 action 覆盖不足限制。

## 1. 交接摘要

| 项目 | 内容 |
|---|---|
| 模型 | GLiClass uni-encoder，`prajjwal1/bert-small` |
| 标签 | verdict 3 类 + action 28 类，共 31 标签 |
| 权重路径 | `model/checkpoints_gpu/final_model_v3/` |
| `model.safetensors` SHA-256 | `f57e3ccd9d22dc86ce5e683dc56d366a166385344de40b82e7e4c3286b2283a0` |
| test 数据 SHA-256 | `c578dea30d67ae1e0a38362632c0a24197a7fb7f76d4552ae0b904adeeb9f857` |
| 推理参数 | `max_length=320`，action 阈值 `0.5` |
| test 样本数 | 5,978 |
| test verdict 分布 | benign 2,293 / suspicious 1,609 / malicious 2,076 |

### 核心结论

| 业务口径 | Precision | Recall | F1 | 解释 |
|---|---:|---:|---:|---|
| 31 标签独立复测 micro | **92.18%** | **92.62%** | **92.40%** | 项目内整体指标 |
| action-only micro | **92.11%** | **92.34%** | **92.22%** | 28 个 action 汇总 |
| action-only macro | **75.10%** | **73.08%** | **73.76%** | 每类等权，暴露长尾短板 |
| 仅 malicious 触发告警 | **99.95%** | **94.75%** | **97.28%** | 误报最低，但 109 条 malicious 未严格判出 |
| suspicious + malicious 触发告警 | **94.80%** | **100%** | **97.33%** | 不漏 malicious，但 benign 告警率约 4.97% |

必须区分指标来源：

- 本报告中的独立复测结果来自 `final_model_v3`。
- `train_pruned.log` 的训练流程 test 指标为 Precision 93.81% / Recall 90.61% / F1 92.18%，与独立复测不是完全相同的评测流程，不能混用。
- `docs/EXTERNAL_EVAL.md` 中的外部公开数据结果是 `final_model_v2` 的冻结基线，不代表 v3；若对外发布 v3 OOD 指标，必须重新运行。

## 2. 指标定义

| 术语 | 白话解释 |
|---|---|
| Precision / 精确率 | 模型报警的样本里，有多少是真的恶意 |
| Recall / 召回率 | 所有真实恶意样本里，有多少被模型抓到 |
| F1 | Precision 和 Recall 的平衡指标 |
| support | 该类别真实样本数 |
| micro | 汇总所有决策后计算；常见类别影响更大 |
| macro | 每个类别等权平均；稀有类别会影响结果 |
| benign FPR | 真实正常命令被误报的比例 |

安全检测通常优先保证 Recall，再根据部署环境控制 Precision 和 FPR。

## 3. 项目内测试：31 标签

独立复测采用 31 标签逐标签统计：

| 指标 | 数值 |
|---|---:|
| TP | 19,508 |
| FP | 1,656 |
| FN | 1,554 |
| Precision | **92.175%** |
| Recall | **92.622%** |
| F1 | **92.398%** |

> 31 标签是 verdict 与 action 的并集，不能直接当作 5,978 条单标签样本的平均准确率。

## 4. Verdict 三分类

### 4.1 混淆矩阵

行是真实类别，列是模型预测类别。

| 真实 \ 预测 | benign | suspicious | malicious |
|---|---:|---:|---:|
| benign | 2,179 | 113 | 1 |
| suspicious | 98 | 1,409 | 102 |
| malicious | 0 | 109 | 1,967 |

三分类准确率：**92.924%**。

### 4.2 每类 Precision / Recall / F1

| 类别 | Precision | Recall | F1 | support |
|---|---:|---:|---:|---:|
| benign | 95.696% | 95.028% | 95.361% | 2,293 |
| suspicious | 86.389% | 87.570% | 86.975% | 1,609 |
| malicious | 95.024% | 94.750% | 94.887% | 2,076 |
| macro 平均 | 92.37% | 92.45% | 92.41% | 5,978 |

`suspicious` 是最弱类别，主要与 `malicious` 混淆。

### 4.3 两种二分类业务口径

严格模式排除真值 suspicious，只比较 benign 与 malicious：

| 策略 | Precision | Recall | F1 | FP | FN | benign FPR |
|---|---:|---:|---:|---:|---:|---:|
| 仅 malicious 报警 | **99.949%** | **94.750%** | **97.280%** | 1 | 109 | **0.0436%** |
| suspicious + malicious 报警 | **94.795%** | **100%** | **97.328%** | 114 | 0 | **4.9717%** |

产品解释：

- 严格模式适合强低误报场景，但会漏掉 109 条真实 malicious。
- 告警模式适合安全运营场景，malicious 不漏；代价是 114/2,293 条 benign 被提示为可疑或恶意。

## 5. Action 指标

action 是多标签任务，一条命令可以有多个 action。因此 support 合计为 15,084，大于 test 样本数 5,978。

### 5.1 汇总

| 指标 | TP | FP | FN | support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| action micro | 13,929 | 1,194 | 1,155 | 15,084 | 92.105% | 92.343% | 92.224% |
| action macro | — | — | — | — | 75.100% | 73.076% | 73.755% |

macro 平均包含无正样本的 `cryptomining`，因此不能单独代表生产效果。

### 5.2 分类别结果

| Action | support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| command_and_control | 2,719 | 98.604% | 98.750% | 98.677% |
| execute_local | 2,427 | 96.757% | 97.116% | 96.936% |
| reverse_shell | 2,393 | 97.317% | 98.537% | 97.924% |
| system_probe | 1,917 | 89.161% | 93.114% | 91.095% |
| file_operation | 1,387 | 94.113% | 95.674% | 94.887% |
| backdoor | 853 | 88.042% | 88.042% | 88.042% |
| bind_shell | 724 | 84.626% | 82.873% | 83.740% |
| download_execute | 405 | 84.419% | 89.630% | 86.946% |
| network_scan | 312 | 75.601% | 70.513% | 72.968% |
| exfiltrate | 311 | 86.420% | 90.032% | 88.189% |
| web_shell | 276 | 71.074% | 62.319% | 66.409% |
| obfuscate | 254 | 81.538% | 62.598% | 70.824% |
| keylog | 239 | 75.688% | 69.038% | 72.210% |
| credential_dump | 180 | 82.011% | 86.111% | 84.011% |
| download | 152 | 90.000% | 88.816% | 89.404% |
| environment_setup | 126 | 86.719% | 88.095% | 87.402% |
| service_persist | 97 | 78.218% | 81.443% | 79.798% |
| disable_security | 69 | 73.214% | 59.420% | 65.600% |
| schedule_persist | 66 | 84.722% | 92.424% | 88.406% |
| clear_logs | 58 | 89.655% | 89.655% | 89.655% |
| account_add | 38 | 67.500% | 71.053% | 69.231% |
| process_inject | 26 | 76.000% | 73.077% | 74.510% |
| self_propagate | 23 | 42.105% | 34.783% | 38.095% |
| registry_persist | 12 | 70.000% | 58.333% | 63.636% |
| timestomp | 11 | 64.286% | 81.818% | 72.000% |
| ransomware | 7 | 75.000% | 42.857% | 54.545% |
| brute_force | 2 | 0% | 0% | 0% |
| cryptomining | 0 | N/A | N/A | N/A |

主要薄弱项：

- `self_propagate`
- `ransomware`
- `disable_security`
- `web_shell`
- `obfuscate`
- `network_scan`
- `keylog`

## 6. 对抗鲁棒性

对抗集目录：`adversarial/`。  
结果目录：`adversarial/results/`。  
完整性校验：`status=verified`、`error_count=0`、模型确认为 `final_model_v3`。

| 项目 | 数量 |
|---|---:|
| 去重候选样本 | 3,620 |
| 全体候选均为攻击意图，无 benign 负样本 | 3,620 |
| 模型判 malicious | 984 |
| 模型判 suspicious | 851 |
| 模型判 benign | 1,785 |

因为对抗集没有 benign 负样本，**不能可靠计算 Precision**，以下只能算 Recall。

| 来源 | 数量 | malicious Recall | suspicious + malicious Recall | benign 静默绕过率 |
|---|---:|---:|---:|---:|
| 子代理生成与变异 | 2,120 | **46.42%** | **86.56%** | 13.44% |
| 320-token 截断模板 | 1,500 | **0%** | **0%** | **100%** |
| 合计 | 3,620 | **27.18%** | **50.69%** | **49.31%** |

结论：

- 普通对抗候选在“suspicious 也告警”的口径下 Recall 为 86.56%，明显好于严格 malicious Recall。
- 320-token 截断模板 Recall 为 0%，是部署层必须修复的真实输入缺口。
- 若强制假设 Precision=100%，严格口径 F1 为 42.75%，告警口径 F1 为 67.29%；这不是可信的生产 F1。

部分弱项：

| 攻击类型 | malicious Recall | 告警 Recall |
|---|---:|---:|
| download | 0% | 93.3% |
| session_chain | 0% | 100% |
| cryptomining | 6.9% | 63.9% |
| ransomware_impact | 12.5% | 79.2% |
| credential_dumping | 13.9% | 94.4% |
| system_probe | 14.0% | 46.0% |

模型经常能判断“可疑”，但没有足够置信度判为 `malicious`。因此 `suspicious` 是重要中间层，不应直接当成良性。

## 7. 外部公开数据 OOD（v2 旧基线）

> 以下结果属于 `final_model_v2`，不能作为 v3 的生产泛化指标，仅用于说明风险方向。

| 数据/子集 | 数量 | 结果 | 指标限制 |
|---|---:|---|---|
| NL2Bash | 10,623 | benign 准确率 81.32%，malicious FPR 0.05%，suspicious 率 18.63% | 无攻击正样本，不能算 Recall |
| GTFOBins 全体 | 810 | malicious Recall 15.06%，非 benign Recall 30.86% | 无良性负样本，不能算 Precision |
| Payloads | 45 | malicious Recall 11.11%，非 benign Recall 26.67% | 提取噪声较多 |
| Cowrie 攻击会话 | 16,517 | malicious 7.0%，suspicious 43.28%，benign 49.72% | 会话内并非每条命令都恶意，不是干净二分类真值 |
| 明确反弹/下载执行/C2 payload | 83 | malicious Recall 75.9%，非 benign Recall 90.4% | 仅正样本 |
| SUID 解释器逃逸 | 79 | malicious Recall 30.4%，非 benign Recall 50.6% | 仅正样本 |
| 双用途命令 | 648 | malicious Recall 5.4%，非 benign Recall 20.8% | 单条命令本身难定性 |

v3 正式 OOD 指标待补：需要用 `final_model_v3` 重跑 `tools/eval_external.py`，并更新 `docs/EXTERNAL_EVAL.md`。

## 8. 风险和待办

| 优先级 | 风险/任务 | 验收指标建议 |
|---|---|---|
| P0 | `max_length=320` 导致恶意动作被截断后完全漏检 | 增加分段扫描；超长样本告警 Recall 不再为 0 |
| P0 | 组合命令/session 语义：单条可能 benign，多条组合恶意 | 建立多命令 session 真值集，报告 session-level P/R/F1 |
| P0 | 对抗集缺少 benign 负样本，无法验证 Precision/FPR | 补齐对抗良性和真实生产良性样本，报告 Precision 与 FPR |
| P1 | 稀有 action 长尾差：self_propagate、ransomware 等 | action macro-F1 从 73.76% 提升；每类 support 足够 |
| P1 | external OOD 仍是 v2 结果 | 用 v3 重跑并冻结版本化结果 |
| P1 | suspicious 与 malicious 边界不稳定 | suspicious F1 从 86.98% 提升，并降低 malicious 漏报 |
| P2 | 单条命令指标不能代表生产告警负担 | 以真实命令流报告每日告警量、FPR 和 MTTR |

推荐的生产验收口径：

| 指标 | 建议最低门槛 |
|---|---:|
| malicious Recall | ≥ 95% |
| malicious vs benign Precision | ≥ 99% |
| suspicious + malicious 告警模式下 benign FPR | 按产品定义，通常 ≤ 1% |
| 超长输入 Recall | ≥ 95%，要求先分段或扩展窗口 |
| action macro-F1 | ≥ 80% |
| session-level Recall | ≥ 90% |

以上是建议门槛，不是当前已达到的结果。

## 9. 复现与结果位置

### 内测

```bash
python -m hyperids.eval \
  --model_dir model/checkpoints_gpu/final_model_v3 \
  --split test
```

注意：`hyperids.eval` 的默认目录仍是 `final_model_v2`，复测 v3 必须显式传入 `--model_dir`。

本轮独立复测使用的临时脚本：

- `/tmp/eval_hyperids_test.py`
- `/tmp/eval_hyperids_actions.py`

### 对抗集

```bash
python adversarial/evaluate_adversarial.py --batch-size 64
python adversarial/verify_results.py
```

关键结果：

- `adversarial/results/summary.json`
- `adversarial/results/all_annotated.jsonl`
- `adversarial/results/verification.json`

### 外部 OOD

```bash
python tools/eval_external.py --batch 256
python tools/analyze_external.py
```

现有结果：

- `dataset/external_eval_results/summary.json`
- `docs/EXTERNAL_EVAL.md`

## 10. 交接结论

1. v3 在项目内 test 分布上约 92% 的 micro F1，常见 verdict 和 action 效果可用。
2. 严格恶意告警几乎无误报，但 Recall 为 94.75%；扩展为 suspicious + malicious 告警后 Recall 达到 100%，代价是 benign 告警率约 4.97%。
3. 真正限制生产泛化的不是单一模型大小，而是长输入截断、session/组合语义、稀有 action 和对抗变体。
4. 对抗集与多数外部公开集缺少完整正负样本，不能据此计算可信 Precision。
5. 下一阶段应先修复输入分段和评测集口径，再用 session-level 数据决定是否升级模型架构。
