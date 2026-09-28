# 训练与复现链路

## 从 corpus 起步（一键）

`dataset/corpus/labeled/` 已通过 Git LFS 保存了 Jev 31 维软标签（verdict 3 + action 28），
从它重建训练集 + 训练 + 导出：

```bash
make data      # corpus → gliclass_v2（31 标签训练集）
make train     # GPU 训练（Lightning L4）
make eval      # 本地评估
make export    # ONNX + INT8
make release   # C 单二进制（静态，权重内嵌）
```

## 完整链路

```text
corpus/labeled/*.jsonl        (Jev 31 维软标签, LFS)
   ↓  build_dataset.build()   (hyperids/build_dataset.py)
dataset/gliclass_v2/          (31 标签 train/val/test)
   ↓  hyperids.train          (GLiClass + bert-small)
model/checkpoints/final_model_v2/
   ↓  deploy/export_onnx.py + quantize_int8.py
model_int8.onnx (30.2MB)
   ↓  deploy/c_runtime/Makefile
hyperids 单二进制 (statically linked)
```

## 训练参数

```bash
python -m hyperids.train \
  --data_dir dataset/gliclass_v2 \
  --save_name final_model_v2 \
  --epochs 4 --batch_size 32 --lr 5e-5
```

最终结果（test 5,978）：verdict_acc 0.917 / action micro-F1 0.909；INT8 基本无损。

## 上游数据准备（可选重跑，见 DATA_PIPELINE.md）

- 公开数据拉取：`hyperids.data.fetch_quasarnix`（TODO 地址）/ `fetch_public` / `fetch_eval`
- LLM 合成：`hyperids.data.synthesize_longtail` / `generate_suid_variants`
- Jev 打标：`hyperids.jev_labels`（31 标签）
