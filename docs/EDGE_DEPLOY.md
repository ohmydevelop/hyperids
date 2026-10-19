# HyperIDs 端侧部署调研与优化结论

目标：把 HyperIDs（29.8M GLiClass / bert-small，31 标签）放进路由器等低端设备，
最好做到 ~10ms/条且不影响业务。以下结论基于当前仓库实测，不执行任何候选命令。

## 1. 是否有“极致优化的开源纯 C 实现”？

**结论：没有现成的、自包含的、面向 GLiClass 的极致优化纯 C 推理实现。**

| 项目 | 语言/依赖 | 结论 |
|---|---|---|
| `Knowledgator/GLiClass.c` | C 调用 ONNX Runtime | 名字像 C，但底层依赖 `onnxruntime-linux-x64`，不是纯 C / 单二进制 |
| `oppiliappan/bert.cpp` / `llama.cpp` 的 BERT 支持 | C++ + ggml，AVX2/NEON/Q4 | 只做 BERT mean-pooling sentence embedding，**没有 GLiClass 的 text/classes projector + dot scorer 头** |
| `nanoE5.c` / `microGPT-c` | 纯 C，4bit 量化 | 证明纯 C 小模型 CPU 可以很低延迟，但不是 GLiClass |
| 本仓库 `deploy/c_runtime/hyperids.c` | 纯 C + AVX2/FMA + per-channel int8 | 最接近需求的纯 C 单二进制实现，本轮已进一步优化 |

因此“有没有”的答案是：**没有可以直接拿来的**；需要自己在 `hyperids.c` 上继续开发。

## 2. 本轮已完成 / 实测

### 2.1 纯 C 运行时优化

- `mm` 从朴素 `s,n,k` 循环改为 **blocked int8 GEMM**（MC=64 / NC=128 / KC=128），
  减少 L1/L2 重复读取；OpenMP 并行粒度也改为 s 块。
- `export_c_labels.py` 修正为与 `hyperids/predict.py` 一致：标签前缀**不含 `[CLS]`**，
  命令后追加 `[SEP]`，并默认指向 `final_model_v9`。
- 修复 `deploy/export_onnx.py` 的输入口径：不再由 tokenizer 自动加 `[CLS]`/尾 `[SEP]`，
  而是 `label_prefix_ids + command_ids + [SEP]` 后 pad，`max_num_classes=31`。
- 修复 `deploy/quantize_int8.py` 对 `.data` 外部权重文件的大小统计。

### 2.2 本机延迟（Intel Core Ultra 7 270K Plus，10 核，AVX2+FMA，KVM）

120 条私有命令，stdin 批处理，`OMP_NUM_THREADS` 分别测试：

| 方案 | 1 线程 | 4 线程 | 8 线程 |
|---|---:|---:|---:|
| 纯 C 单二进制（blocked int8，静态） | ~166 ms/条 | **~74 ms/条** | ~106 ms/条 |
| ONNX Runtime fp32（seq=256） | ~40 ms/条 | ~26 ms/条 | ~23 ms/条 |
| ONNX Runtime INT8（seq=256） | **~12.7 ms/条** | ~17 ms/条 | **~9.9 ms/条** |

说明：
- 纯 C 的 8 线程反而慢，是短序列 GEMM 并行开销 + 内存带宽问题，不表示 8 核无用；
- ORT INT8 能达到 10ms 级别，但依赖 ONNX Runtime，不是单二进制。

### 2.3 对齐性

- ONNX fp32 与 PyTorch fp32：logits max abs diff < 3e-5，verdict 一致。
- ONNX INT8 与 PyTorch fp32：max abs diff 约 1–2 logits，verdict 在抽测样本一致。
- 纯 C int8 与 PyTorch fp32（80 条私有冻结集）：
  - verdict 80/80；
  - action 76/80，剩余 4 条是 `action.system_probe` 在 0.50 阈值附近由 int8 量化造成的边界翻转。

## 3. 能否在低端路由器做到 10ms？

**用当前模型直接硬优化：基本不现实。**

原因：每条命令前向约 2.8B MACs；10ms 需要约 560 GFLOPS 有效算力。
低端路由器常见 CPU 是 ARM Cortex-A7/A53、MIPS 单/双核，理论峰值多在几到 20 GFLOPS，
实际 GEMM 利用率 30–60% 时有效算力更低。当前 VM 上有 AVX2 纯 C 也要 ~74ms，低端设备只会更慢。

### 可行路径（按推荐顺序）

1. **先蒸馏 / 换小模型（首选）**
   训练一个 2–5M 参数、命令-only 的分类器（或 gliclass-edge 系列蒸馏），
   输入 token 从 193（31 个标签前缀）降到几十 token，计算量可下降 1–2 个数量级，
   这才是在路由器上做 10ms 的关键。不要把优化都押在 GEMM 上。

2. **ONNX Runtime + XNNPACK/oneDNN + 动态 INT8 + 受限 seq**
   最快落地；ARM 路由器可用 XNNPACK。代价是运行时依赖和内存占用约 96MB，
   不是单二进制，也不完全符合“纯 C”。
   `seq_len=256` 时命令部分最多约 62 token；长命令需要 seq=512 + 滑动窗口，
   否则存在截断。

3. **纯 C 单二进制继续优化（保持无依赖）**
   当前 `hyperids.c` 已支持非 x86 标量 fallback（去掉 AVX2 强依赖），可在 ARM/MIPS 交叉编译；
   但没有 NEON kernel 时速度会很慢。后续方向：
   - ARM NEON `int8 -> int32` 点积 kernel（`vdotq_s32` / `vmull`）；
   - 权重预打包成更适合 cache 的布局（N 块 / K 块）；
   - 用 ggml Q4_0 或 Q4_K 进一步压缩 17.6MB → 5MB 级；
   - 固定 seq=256 并去掉 512 的 workspace，降低峰值内存；
   - 对多头 attention 做按 head 的 kernel 化，避免 `nn*nn` 中间大矩阵。
   即使做完，当前 29.8M 模型在低端单核上也不保证 10ms。

## 4. 推荐结论

- **当前生产部署**：服务器/网关用 `deploy/c_runtime/build_omp/hyperids` 静态二进制，
  或 ONNX Runtime INT8；两者都无 Python 依赖（C 二进制无任何依赖）。
- **路由器等低端端侧**：先做命令-only 小模型蒸馏，再用纯 C/ggml/NEON 部署；
  不建议在当前 29.8M + 193-token 标签前缀模型上追求 10ms。

## 5. 安全声明

本调研只读取文本与权重，不执行、不 `source`、不 `eval` 任何候选 shell 命令。
所有命令样例仅用于离线模型推理与延迟统计。
