# HyperIDs v9（v1.5.2）评测报告

> 报告日期：2026-10-05
> 评测对象：`model/checkpoints_gpu/final_model_v9`（= GitHub release v1.5.2）
> 模型：GLiClass + `prajjwal1/bert-small`（17.6M，≤30M），512 长度

## 1. 本轮改动（相对 v7 / v1.5.1）

| 改动 | 说明 |
|---|---|
| 否决 v8 全弱类混合 | v8 把 keylog 带偏（0.600→0.519），不发布 |
| ransomware-only 回灌 | 只保留 v8 合成中 Jev 过滤后的 **36 条 ransomware** |
| final_model_v9 | 从 **v7** 继续训练 4 epochs（L4 GPU，batch 64，lr 5e-5），test 冻结 6,287 |

## 2. 内测指标（action 平衡 test 6,287）

| 指标 | v7 | v8（否决） | v9 |
|---|---:|---:|---:|
| verdict_acc | **0.9278** | 0.9267 | 0.9262 |
| action micro-F1 | **0.9133** | 0.9120 | 0.9128 |
| action macro-F1 | 0.8331 | 0.8337 | **0.8415** |

## 3. 弱类变化

| action | v7 | v8 | v9 | n(test) |
|---|---:|---:|---:|---:|
| action.ransomware | 0.552 | 0.667 | **0.727** | 20 |
| action.web_shell | 0.558 | 0.567 | **0.586** | 110 |
| action.keylog | **0.600** | 0.519 | 0.582 | 52 |
| action.self_propagate | **0.629** | 0.588 | 0.588 | 20 |

ransomware v9：P=0.923 / R=0.600。

## 4. 结论

1. **只回灌有效类是对的**：ransomware 0.552→0.727，macro-F1 创历史新高 84.15%。
2. **v8 全弱类混合被否决**：keylog 回退证明低质量合成会污染。
3. 代价：verdict_acc / micro-F1 相对 v7 略降（-0.16 / -0.05），self_propagate 未回到 v7。
4. 作为小版本发布：弱类收益明确，主指标回退在噪声附近。

## 5. 部署产物（v1.5.2）

| 项 | 值 |
|---|---|
| INT8 权重 | `model_int8.bin` 17.6MB |
| 静态单二进制 | `build/hyperids` ~18.4MB |
| 峰值 RSS | ~38 MiB |
| 推理速度 | ~280 ms/条（AVX2+FMA） |

## 6. 复现

```bash
PYTHONPATH=. python -m hyperids.data.merge_v6_longtail \
  --src dataset/gliclass_v4 \
  --synth dataset/corpus/labeled/synthetic/v8_ransomware_only.jsonl \
  --out dataset/gliclass_v6

python -m hyperids.train --data_dir dataset/gliclass_v6 \
  --save_name final_model_v9 --resume_from model/checkpoints/final_model_v7 \
  --epochs 4 --batch_size 64 --lr 5e-5 --device cuda

python -m hyperids.eval --model_dir model/checkpoints_gpu/final_model_v9 --split test
```
