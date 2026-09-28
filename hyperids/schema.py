"""HyperIDs current label schema: verdict (3) + action (28) = 31 labels.

Source of truth: label_schema.yaml
Legacy 199-label schema -> current 31-label mapping lives in the `from_old` field.
MITRE tactic/technique are NOT predicted; they are derived from actions via
`action_attck()` (a deterministic rule table).
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
def action_from_old() -> dict[str, tuple[str, ...]]:
    """action id -> old-schema source label ids (sum their soft probs)."""
    return {x["id"]: tuple(x["from_old"]) for x in load()["action"]}


@lru_cache(maxsize=1)
def verdict_from_old() -> dict[str, str]:
    """verdict id -> old risk id (1:1)."""
    return {
        "verdict.benign": "risk.benign",
        "verdict.suspicious": "risk.suspicious",
        "verdict.malicious": "risk.malicious",
    }


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
