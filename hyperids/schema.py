"""HyperIDs 标签 schema：verdict（3）+ action（28）= 31 标签。

Source of truth: label_schema.yaml
MITRE tactic/technique 不预测，由 actions 经 `action_attck()` 规则表确定性推导。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

SCHEMA_PATH = Path(__file__).resolve().parent / "label_schema.yaml"


@lru_cache(maxsize=1)
def load() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def verdict_ids() -> tuple[str, ...]:
    return tuple(x["id"] for x in load()["verdict"])


@lru_cache(maxsize=1)
def action_ids() -> tuple[str, ...]:
    return tuple(x["id"] for x in load()["action"])


@lru_cache(maxsize=1)
def all_label_ids() -> tuple[str, ...]:
    return verdict_ids() + action_ids()


@lru_cache(maxsize=1)
def group_offsets() -> dict[str, tuple[int, int]]:
    nv = len(verdict_ids())
    na = len(action_ids())
    return {"verdict": (0, nv), "action": (nv, nv + na)}


@lru_cache(maxsize=1)
def descriptions() -> dict[str, str]:
    d = load()
    out = {x["id"]: x["description"] for x in d["verdict"]}
    out.update({x["id"]: x["description"] for x in d["action"]})
    return out


@lru_cache(maxsize=1)
def action_attck() -> dict[str, dict]:
    """action id -> {'tactics':[...], 'techniques':[...]} (deterministic rules)."""
    return {x["id"]: {"tactics": x["tactics"], "techniques": x["techniques"]} for x in load()["action"]}


def derive_attck(actions) -> dict:
    """Given a set of action ids, derive tactic + technique sets via rules."""
    rules = action_attck()
    tactics, techniques = set(), set()
    for a in actions:
        r = rules.get(a)
        if r:
            tactics.update(r["tactics"])
            techniques.update(r["techniques"])
    return {"tactics": sorted(tactics), "techniques": sorted(techniques)}


if __name__ == "__main__":
    print("verdict:", verdict_ids())
    print("actions:", len(action_ids()))
    print("total labels:", len(all_label_ids()))
    print("group offsets:", group_offsets())
