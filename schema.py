"""Shared label schema loader — single source of truth for Teacher/Student/RedTeam."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import yaml

SCHEMA_PATH = Path(__file__).resolve().parent / "label_schema.yaml"
GROUPS = ("risk", "intent", "tactic", "technique")


@lru_cache(maxsize=1)
def load() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def all_label_ids() -> tuple[str, ...]:
    """Flat ordered list of every label id — Student scoring dimension."""
    d = load()
    return tuple(x["id"] for g in GROUPS for x in d[g])


@lru_cache(maxsize=1)
def label_descriptions() -> dict[str, str]:
    """id → description, for GLiClass label representations."""
    d = load()
    return {x["id"]: x["description"] for g in GROUPS for x in d[g]}


@lru_cache(maxsize=1)
def group_offsets() -> dict[str, tuple[int, int]]:
    """group → (start, end) slice into all_label_ids()."""
    d = load()
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for g in GROUPS:
        n = len(d[g])
        offsets[g] = (cursor, cursor + n)
        cursor += n
    return offsets


def num_labels() -> int:
    return len(all_label_ids())


# --- Collapsed technique schema (child .NNN -> parent) ---

def _technique_parent_id(tech_id: str) -> str:
    """technique.T1059.004 -> technique.T1059 (MITRE sub-technique collapse)."""
    return re.sub(r"\.\d{3}$", "", tech_id)


@lru_cache(maxsize=1)
def technique_parent_map() -> dict[str, str]:
    """child technique id -> parent technique id (only .NNN entries map)."""
    tech_ids = all_label_ids()[group_offsets()["technique"][0]: group_offsets()["technique"][1]]
    return {t: _technique_parent_id(t) for t in tech_ids if _technique_parent_id(t) != t}


@lru_cache(maxsize=1)
def collapsed_label_ids() -> tuple[str, ...]:
    """132-label schema: risk 3 + intent 41 + tactic 14 + technique parent 74."""
    off = group_offsets()
    risk = all_label_ids()[off["risk"][0]: off["risk"][1]]
    intent = all_label_ids()[off["intent"][0]: off["intent"][1]]
    tactic = all_label_ids()[off["tactic"][0]: off["tactic"][1]]
    tech = all_label_ids()[off["technique"][0]: off["technique"][1]]
    parents = list(dict.fromkeys(_technique_parent_id(t) for t in tech))  # keep first-seen order
    return tuple(risk + intent + tactic + tuple(parents))


@lru_cache(maxsize=1)
def collapsed_group_offsets() -> dict[str, tuple[int, int]]:
    """group -> (start, end) into collapsed_label_ids()."""
    off = group_offsets()
    n_risk = off["risk"][1] - off["risk"][0]
    n_intent = off["intent"][1] - off["intent"][0]
    n_tactic = off["tactic"][1] - off["tactic"][0]
    n_tech = len(collapsed_label_ids()) - n_risk - n_intent - n_tactic
    risk = (0, n_risk)
    intent = (n_risk, n_risk + n_intent)
    tactic = (n_risk + n_intent, n_risk + n_intent + n_tactic)
    technique = (n_risk + n_intent + n_tactic, n_risk + n_intent + n_tactic + n_tech)
    return {"risk": risk, "intent": intent, "tactic": tactic, "technique": technique}


def collapse_labels(labels) -> set[str]:
    """Map a set/list of full-schema labels to the collapsed 132-label schema."""
    parent = technique_parent_map()
    out = set()
    for l in labels:
        out.add(parent.get(l, l))
    return out


if __name__ == "__main__":
    print(f"version : {load()['version']}")
    print(f"labels  : {num_labels()}")
    for g, (s, e) in group_offsets().items():
        print(f"  {g:12s} [{s:3d}:{e:3d})  n={e-s}")
