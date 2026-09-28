# 训练与复现链路

## 从 corpus 起步（推荐，一键）

`dataset/corpus/labeled/` 已通过 Git LFS 保存了 Jev 199 维软标签（最贵资产），
从它重建训练集 + 训练 + 导出：

```bash
make data      # corpus → gliclass_v2（31 标签训练集）
make train     # GPU 训练（Lightning L4）
make eval      # 本地评估
make export    # ONNX + INT8
make release   # C 单二进制（静态，权重内嵌）
```

## 完整链路（6 步）

```text
corpus/labeled/*.jsonl        (Jev 199 维软标签, LFS)
   ↓  merge_corpus_to_soft    (hyperids/rebuild_from_corpus.py)
dataset/soft_labels_50k.jsonl
   ↓  prepare_data.build()     (hyperids/prepare_data.py)
dataset/gliclass/             (199 标签 train/val/test + split)
   ↓  collapse.main()          (hyperids/collapse.py)
dataset/gliclass_collapsed/   (132 标签 split)
   ↓  build_dataset.main()     (hyperids/build_dataset.py, 199→31 映射)
dataset/gliclass_v2/          (31 标签: verdict 3 + action 28)
   ↓  hyperids.train           (GLiClass + bert-small)
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
- LLM 合成：`hyperids.data.bulk_50k` / `synthesize_longtail` / `generate_suid_variants`
- Jev 打标：`hyperids.jev_labels_legacy`(199) / `hyperids.jev_labels`(31)
