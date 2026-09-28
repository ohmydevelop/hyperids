#!/usr/bin/env python3
"""校验 label_schema.yaml（31 标签：verdict 3 + action 28）。"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "hyperids" / "label_schema.yaml"
GROUPS = ("verdict", "action")


def load() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check(data: dict) -> list[str]:
    errors: list[str] = []
    for g in GROUPS:
        if g not in data or not isinstance(data[g], list) or not data[g]:
            errors.append(f"missing or empty group: {g}")
    if errors:
        return errors

    for g in GROUPS:
        for item in data[g]:
            iid = item.get("id", "<missing-id>")
            if "id" not in item:
                errors.append(f"{g}: entry missing id")
            if "description" not in item or not str(item["description"]).strip():
                errors.append(f"{iid}: missing description")
            if iid and not iid.startswith(f"{g}."):
                errors.append(f"{iid}: id must start with '{g}.'")

    all_ids = [x["id"] for g in GROUPS for x in data[g] if "id" in x]
    for d, c in Counter(all_ids).items():
        if c > 1:
            errors.append(f"duplicate id: {d}")

    # action 必须有 tactics/techniques（供 derive_attck 推导）
    for a in data.get("action", []):
        if not a.get("tactics") or not a.get("techniques"):
            errors.append(f"{a.get('id')}: action missing tactics/techniques")

    counts = data.get("counts", {})
    for g in GROUPS:
        expected, actual = counts.get(g), len(data[g])
        if expected is not None and expected != actual:
            errors.append(f"counts.{g}: declared {expected}, actual {actual}")
    return errors


def main() -> int:
    data = load()
    errors = check(data)
    print(f"schema version: {data.get('version')}")
    for g in GROUPS:
        print(f"  {g:10s} {len(data[g]):4d}")
    if errors:
        print(f"\nFAIL — {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nPASS — label_schema.yaml is valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
