# 部署（最终 v2）

最终模型 v2：`prajjwal1/bert-small`（29.8M）+ 31 标签（verdict 3 + action 28）。
MITRE tactic/technique 由 `schema_v2.derive_attck()` 规则表推导，不在模型中预测。

## RSS / 体积（实测）

| 运行时 | 权重 | 峰值 RSS |
|---|---|---|
| ONNX Runtime C | fp32（118MB external data） | 114.7MB |
| ONNX Runtime C | **INT8（30.2MB）** | **96.2MB ✅** |

- INT8 与 fp32 几乎无损：verdict_acc 0.9187（fp32 0.9167）、action micro-F1 0.9099（fp32 0.909）。
- C 侧已关 `ORT_ENABLE_ALL` 图优化、`MemPattern`、`CpuMemArena` 压低峰值。

## 产物

- `model/checkpoints_gpu/final_model_v2/model.onnx` —— fp32（external data）
- `model/checkpoints_gpu/final_model_v2/model_int8.onnx` —— INT8（单文件）
- `export/c_infer_example_v2.c` —— C 推理示例（N_LOGITS=31, SEQ_LEN=320）
- `export/prepare_c_input.py` —— 命令+标签 tokenize 成 int64 .bin（当前为 edge 版，v2 用 `eval_v2.py` 同款拼接即可）

## 使用流程

```bash
# 1) 导出 ONNX（31 标签一次前向）
python -m export.export_onnx --model_dir model/checkpoints_gpu/final_model_v2 \
    --out model/checkpoints_gpu/final_model_v2/model.onnx --chunk 31 --seq_len 320 --collapsed 2>/dev/null || true

# 2) INT8 量化
python export/quantize_int8_onnx.py --in model/checkpoints_gpu/final_model_v2/model.onnx \
    --out model/checkpoints_gpu/final_model_v2/model_int8.onnx

# 3) C 推理（编译参考 export/c_infer_example_v2.c）
./export/c_infer_example_v2 model/checkpoints_gpu/final_model_v2/model_int8.onnx \
    export/input_ids_v2.bin export/attention_mask_v2.bin
```

## 推理约定

31 标签一次前向输出 `[1,31]` logits：前 3 个是 verdict（argmax），后 28 个是 action
（阈值 0.5）。拿到 action 集合后调用 `schema_v2.derive_attck(actions)` 得到 tactic/technique。

## 历史备注

- 早期 electra-13.75M（199 标签）、edge-32.7M（199 标签）、以及 132 标签折叠版的
  探索过程记录在 `PROGRESS.md`；这些不是最终交付物，仅作对照保留。
- `export/c_infer_example.c`（edge）、`c_infer_example_bert.c`（132 标签）为历史版本。
