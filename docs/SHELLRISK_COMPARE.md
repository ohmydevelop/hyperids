# ShellRisk-Bench 对比：Kestrel vs HyperIDs v9

评测日期：2026-10-19
数据集：`kontext-security/ShellRisk-Bench` test split（公开）
样本：4194 条，risky=193，not_risky=4001

## 1. 二分类效果

| 模型 / 口径 | Precision | Recall | F1 | Accuracy | FPR |
|---|---:|---:|---:|---:|---:|
| Kestrel thr=0.0 | 0.9897 | **1.0000** | 0.9948 | 0.9995 | 0.000500 |
| Kestrel thr=0.40 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.000000** |
| HyperIDs strict（malicious） | 0.0788 | 0.0674 | 0.0726 | 0.9208 | 0.037991 |
| HyperIDs loose（malicious+suspicious） | 0.0503 | 0.8394 | 0.0949 | 0.2635 | 0.764309 |

混淆矩阵（TP/FP/FN/TN）：

| 模型 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| Kestrel thr=0.0 | 193 | 2 | 0 | 3999 |
| Kestrel thr=0.40 | 193 | 0 | 0 | 4001 |
| HyperIDs strict | 13 | 152 | 180 | 3849 |
| HyperIDs loose | 162 | 3058 | 31 | 943 |

## 2. 性能

ShellRisk test 全量评分耗时：

| 模型 | 全量 4194 条 | 单条延迟 |
|---|---:|---:|
| Kestrel Python 参考实现 | 0.145 s | ~0.0556 ms/条 |
| HyperIDs v9 PyTorch CPU 4 线程 | 124.613 s | ~22.646 ms/条 |

额外部署口径（本机历史实测）：

| 部署形态 | 单条延迟 |
|---|---:|
| HyperIDs C 静态单二进制，4 线程 | ~74 ms/条 |
| HyperIDs Rust + ONNX Runtime INT8，静态内嵌，8 线程 | ~8.1 ms/条 |

## 3. 结论

1. **在 ShellRisk-Bench 上，Kestrel 全面占优**。
   `thr=0.40` 在该 test split 上达到 Precision/Recall/F1 = 1.0，FPR = 0。

2. **但这有基准分布因素**。
   ShellRisk-Bench 是 Kestrel 的官方 benchmark，训练/测试同源，Kestrel 对它做过校准。
   因此不能把该结果直接外推为“Kestrel 在任何生产 shell 风险场景都完美”。

3. **HyperIDs v9 不适配这个 benchmark 的语义**。
   - ShellRisk 的 `risky` 主要是 agent tool-call 风险，而不是“明确恶意 shell 攻击”；
   - HyperIDs v9 的 strict malicious 只召回 6.74%，大量 risky 被判 benign；
   - HyperIDs loose 召回 83.94%，但 FPR 高达 76.43%，说明它对 SWE/终端仿真等正常命令过度报警。

4. **两者定位不同**。
   - Kestrel：二分类 risk gate，轻量、快、适合 ShellRisk 风格的命令风险拦截。
   - HyperIDs：3-class verdict + 28 action + MITRE，适合需要可解释威胁分类的场景。

5. **可组合使用**。
   如果生产场景同时需要“低 FPR 二分类门禁”和“恶意命令解释”，可以用 Kestrel 做第一道 binary gate，HyperIDs 只对命中或需要解释的命令输出 action/MITRE。

## 4. 复现

数据与脚本：

```text
dataset: /tmp/shellrisk/data/test-00000-of-00001.parquet
Kestrel: kontext-security/Kestrel classifier.json
HyperIDs: model/checkpoints_gpu/final_model_v9
```

评测输出：

```text
/tmp/shellrisk_compare.json
```

脚本见 `tools/eval_shellrisk.py`（公开数据，不读 private，不执行命令）。
