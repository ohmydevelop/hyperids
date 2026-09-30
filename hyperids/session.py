"""Session 级检测（规则聚合）：单命令推理 + 攻击链规则。

不改模型，对 session（多条独立命令序列）逐条推理，再按攻击链规则聚合，
输出 session 级 verdict。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from hyperids.predict import Predictor
from hyperids import schema

# 攻击链规则：基于 session 内 action 的时序/战术组合
# 每条规则 = (名称, 需要同时出现的 action 集合)
ATTACK_CHAINS = [
    ("download_execute", {"action.download", "action.execute_local"}),
    ("download_execute", {"action.download", "action.download_execute"}),
    ("recon_priv_exfil",
     {("action.system_probe", "action.network_scan"), "action.process_inject", "action.exfiltrate"}),
    ("recon_priv_exfil",
     {"action.system_probe", "action.account_add", "action.exfiltrate"}),
    ("persist_clearlog",
     {"action.schedule_persist", "action.clear_logs"}),
    ("persist_clearlog",
     {"action.service_persist", "action.clear_logs"}),
    ("persist_clearlog",
     {"action.schedule_persist", "action.timestomp"}),
    ("c2_persist",
     {"action.reverse_shell", "action.service_persist"}),
    ("c2_persist",
     {"action.reverse_shell", "action.schedule_persist"}),
    ("c2_persist",
     {"action.reverse_shell", "action.registry_persist"}),
]

VERDICT_RANK = {"verdict.benign": 0, "verdict.suspicious": 1, "verdict.malicious": 2}


@dataclass
class SessionResult:
    session_id: str
    verdict: str                     # 聚合后的 session verdict
    per_command_verdicts: list[str]  # 每条命令的 verdict
    hit_chains: list[str] = field(default_factory=list)
    all_actions: set[str] = field(default_factory=set)


def _match_chains(actions: set[str]) -> list[str]:
    """返回命中的攻击链名。"""
    hits = []
    for name, needed in ATTACK_CHAINS:
        # needed 里可能含「或」组（tuple），只要组内任一命中即可
        ok = True
        for item in needed:
            if isinstance(item, tuple):
                if not any(a in actions for a in item):
                    ok = False
                    break
            elif item not in actions:
                ok = False
                break
        if ok:
            hits.append(name)
    return list(dict.fromkeys(hits))  # 去重保序


def score_session(predictor: Predictor, session_id: str, commands: list[str]) -> SessionResult:
    """单命令逐条推理 + 规则聚合。"""
    per_verdicts = []
    all_actions: set[str] = set()
    for cmd in commands:
        r = predictor.predict(cmd)
        per_verdicts.append(r["verdict"])
        all_actions.update(r["actions"])

    hit_chains = _match_chains(all_actions)

    # 聚合 verdict：任一恶意 -> malicious；命中攻击链 -> 至少 suspicious（提级）；否则 max
    rank = max(VERDICT_RANK[v] for v in per_verdicts)
    if "verdict.malicious" in per_verdicts:
        verdict = "verdict.malicious"
    elif hit_chains:
        # 命中攻击链但单命令都是 benign/suspicious -> 提级到 suspicious 告警
        verdict = "verdict.suspicious"
    else:
        verdict = max(per_verdicts, key=lambda v: VERDICT_RANK[v])
    return SessionResult(
        session_id=session_id,
        verdict=verdict,
        per_command_verdicts=per_verdicts,
        hit_chains=hit_chains,
        all_actions=all_actions,
    )


if __name__ == "__main__":
    import json
    from pathlib import Path
    p = Predictor()
    rows = [json.loads(l) for l in Path("adversarial/raw/session_sequences.jsonl").read_text().splitlines() if l.strip()]
    for r in rows[:5]:
        sr = score_session(p, r["session_id"], r["commands"])
        print(f"{sr.session_id:22s} gt={r['ground_truth']:9s} -> {sr.verdict:20s} chains={sr.hit_chains}")
