# HyperIDs 顶层入口 —— 从 corpus 到可用模型 + 部署
.PHONY: data train eval export release rebuild clean

# 1) 从 corpus（Jev 199 软标签，LFS 已同步）重建 31 标签训练集
data:
	python -m hyperids.rebuild_from_corpus

# 2) GPU 训练（Lightning L4 编排，见 scripts/）
train:
	python scripts/run_full_training.py

# 3) 本地评估（需先下载 checkpoint 到 model/checkpoints_gpu/）
eval:
	python -m hyperids.eval --model_dir model/checkpoints_gpu/final_model_v2 --split test

# 4) 导出 ONNX + INT8 量化
export:
	python deploy/export_onnx.py --model_dir model/checkpoints_gpu/final_model_v2 \
		--out model/checkpoints_gpu/final_model_v2/model.onnx --chunk 31 --seq_len 320
	python deploy/quantize_int8.py --in model/checkpoints_gpu/final_model_v2/model.onnx \
		--out model/checkpoints_gpu/final_model_v2/model_int8.onnx

# 5) 编译 C 单二进制（静态，模型权重内嵌）
release:
	cd deploy/c_runtime && make

rebuild: data train

clean:
	rm -rf deploy/c_runtime/build deploy/c_runtime/ci_assets
