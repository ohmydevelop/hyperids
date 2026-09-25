"""Shared label schema loader — single source of truth for Teacher/Student/RedTeam."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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


if __name__ == "__main__":
    print(f"version : {load()['version']}")
    print(f"labels  : {num_labels()}")
    for g, (s, e) in group_offsets().items():
        print(f"  {g:12s} [{s:3d}:{e:3d})  n={e-s}")
