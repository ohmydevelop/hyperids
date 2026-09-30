# HyperIDs v5（v1.4.0）评测报告

> 报告日期：2026-09-30
> 评测对象：`model/checkpoints_gpu/final_model_v5`（= GitHub release v1.4.0）
> 模型：GLiClass + `prajjwal1/bert-small`，词表裁剪后 17.6M，512 长度

## 1. 本轮改动（相对 v4 / v1.3.0）

| 改动 | 说明 |
|---|---|
| action 平衡 test 集 | `gliclass_v2` 重切分：28 个 action 每类 test ≥ 20 条，test 5,978 → 6,287 |
| 继续训练 | `--resume_from final_model_v4`，4 epochs / batch 64 / lr 5e-5 / L4 GPU |
| 修复不可测类 | 原 test 集 cryptomining=0 / brute_force=2 / ransomware=7，无法真实评估 |

> 之前 v4 的 action macro-F1 73.76% 是「部分类别 test 无样本」造成的虚高/虚低并存；
> 本轮在同一份 action 平衡 test 集上对比 v4 与 v5，才是可比的真实提升。

## 2. 内测指标（action 平衡 test 6,287，v4 vs v5 同一 test 集）

| 指标 | v4 | v5 | Δ |
|---|---:|---:|---:|
| verdict_acc | 0.9289 | **0.9310** | +0.0021 |
| action micro-Precision | 0.8021 | **0.9092** | +0.1071 |
| action micro-Recall | 0.9362 | **0.9140** | -0.0222 |
| action micro-F1 | 0.8640 | **0.9116** | +0.0476 |
| action macro-F1 | 0.6704 | **0.8052** | **+0.1348** |

> 阈值统一 0.5。micro 是 28 个 action 的全局多标签 P/R/F1；macro 是 28 类 F1 的简单平均。

## 3. per-action F1（阈值 0.5，n=该类 test 样本数）

| action | v4 F1 | v5 F1 | n |
|---|---:|---:|---:|
| action.download | 0.802 | **0.893** | 100 |
| action.execute_local | 0.895 | **0.925** | 2140 |
| action.download_execute | 0.752 | **0.833** | 285 |
| action.command_and_control | 0.959 | **0.971** | 2572 |
| action.obfuscate | 0.695 | **0.751** | 151 |
| action.file_operation | 0.871 | **0.908** | 1252 |
| action.process_inject | 0.480 | **0.793** | 29 |
| action.network_scan | 0.724 | **0.832** | 159 |
| action.brute_force | 0.000 | **0.840** | 26 |
| action.reverse_shell | 0.913 | **0.943** | 2045 |
| action.bind_shell | 0.694 | **0.769** | 348 |
| action.web_shell | 0.518 | **0.593** | 110 |
| action.backdoor | 0.794 | **0.871** | 600 |
| action.keylog | 0.306 | **0.500** | 52 |
| action.credential_dump | 0.837 | **0.888** | 186 |
| action.ransomware | 0.320 | **0.462** | 20 |
| action.cryptomining | 0.000 | **0.919** | 20 |
| action.exfiltrate | 0.801 | **0.846** | 287 |
| action.disable_security | 0.621 | **0.788** | 37 |
| action.clear_logs | 0.860 | **0.901** | 56 |
| action.timestomp | 0.783 | **0.833** | 24 |
| action.account_add | 0.807 | **0.891** | 48 |
| action.registry_persist | 0.894 | **0.917** | 25 |
| action.service_persist | 0.636 | **0.797** | 62 |
| action.schedule_persist | 0.827 | **0.844** | 67 |
| action.self_propagate | 0.424 | **0.308** | 20 |
| action.system_probe | 0.889 | **0.912** | 1979 |
| action.environment_setup | 0.669 | **0.817** | 113 |

## 4. 核心结论

1. **宏观显著提升**：action macro-F1 0.6704 → 0.8052（+13.5 点），micro-F1 0.8640 → 0.9116。
2. **两个 0 分长尾类被救活**：brute_force 0.000 → 0.840、cryptomining 0.000 → 0.919。
3. **其余长尾类整体上行**：process_inject +0.313、disable_security +0.167、network_scan +0.108、
   web_shell +0.075、ransomware +0.142、keylog +0.194。
4. **仍在拉低 macro 的类**：self_propagate（0.308，v4 反而 0.424，需单独查）、web_shell（0.593）、
   ransomware（0.462）、keylog（0.500）——这些是下一步补数据/阈值优化的重点。
5. **注意**：micro-Recall 从 0.9362 略降到 0.9140，precision 从 0.8021 升到 0.9092——
   v5 更「审慎」：少报了一些长尾，但报得准得多（precision +10.7 点）。

## 5. 部署产物（v1.4.0）

| 项 | 值 |
|---|---|
| INT8 权重 | `model_int8.bin` 17.6MB（per-channel int8，2D 量化 / 1D fp32） |
| 静态单二进制 | `build/hyperids` ~18.4MB（statically linked） |
| 峰值 RSS | ~38 MiB（<100MB 目标） |
| 推理速度 | ~280 ms/条（AVX2+FMA，CPU，x86-64-v3） |

## 6. 复现

```bash
# 训练（GPU，L4）
python -m hyperids.train --data_dir dataset/gliclass_v2 \
  --save_name final_model_v5 --resume_from model/checkpoints_gpu/final_model_v4 \
  --epochs 4 --batch_size 64 --lr 5e-5 --device cuda

# 评估（action 平衡 test）
python -m hyperids.eval --model_dir model/checkpoints_gpu/final_model_v5 --split test

# 导出 INT8 权重 + 编译 C 单二进制
PYTHONPATH=. python deploy/c_runtime/export_c_weights.py
PYTHONPATH=. python deploy/c_runtime/export_c_labels.py
cd deploy/c_runtime && make
```
