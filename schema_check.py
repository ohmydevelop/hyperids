#!/usr/bin/env python3
"""Validate label_schema.yaml — contract check for Teacher / Student / Red Team."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

SCHEMA_PATH = Path(__file__).resolve().parent / "label_schema.yaml"
GROUPS = ("risk", "intent", "tactic", "technique")


def load() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check(data: dict) -> list[str]:
    errors: list[str] = []

    # 1. required groups present
    for g in GROUPS:
        if g not in data or not isinstance(data[g], list) or not data[g]:
            errors.append(f"missing or empty group: {g}")

    if errors:
        return errors

    # 2. each entry has id + description
    for g in GROUPS:
        for item in data[g]:
            iid = item.get("id", "<missing-id>")
            if "id" not in item:
                errors.append(f"{g}: entry missing id")
            if "description" not in item or not str(item["description"]).strip():
                errors.append(f"{iid}: missing description")

    # 3. unique ids across all groups
    all_ids = [x["id"] for g in GROUPS for x in data[g] if "id" in x]
    dups = [i for i, c in Counter(all_ids).items() if c > 1]
    for d in dups:
        errors.append(f"duplicate id: {d}")

    # 4. prefix matches group
    for g in GROUPS:
        for item in data[g]:
            iid = item.get("id", "")
            if iid and not iid.startswith(f"{g}."):
                errors.append(f"{iid}: id must start with '{g}.'")

    # 5. technique.tactic references a known tactic
    tactic_ids = {t["id"] for t in data["tactic"]}
    for t in data.get("technique", []):
        ref = t.get("tactic")
        if not ref:
            errors.append(f"{t.get('id')}: technique missing tactic ref")
        elif ref not in tactic_ids:
            errors.append(f"{t.get('id')}: unknown tactic ref '{ref}'")

    # 6. technique extra fields
    for t in data.get("technique", []):
        for field in ("mitre_id", "name", "description", "commands_hint"):
            if not t.get(field):
                errors.append(f"{t.get('id')}: missing {field}")

    # 7. counts block matches reality
    counts = data.get("counts", {})
    for g in GROUPS:
        expected = counts.get(g)
        actual = len(data[g])
        if expected is not None and expected != actual:
            errors.append(f"counts.{g}: declared {expected}, actual {actual}")
    total_declared = counts.get("total_labels")
    total_actual = sum(len(data[g]) for g in GROUPS)
    if total_declared is not None and total_declared != total_actual:
        errors.append(f"counts.total_labels: declared {total_declared}, actual {total_actual}")

    # 8. version present
    if not data.get("version"):
        errors.append("missing version")

    return errors


def main() -> int:
    data = load()
    errors = check(data)
    print(f"schema version: {data.get('version')}")
    for g in GROUPS:
        print(f"  {g:12s} {len(data[g]):4d}")
    print(f"  {'TOTAL':12s} {sum(len(data[g]) for g in GROUPS):4d}")
    if errors:
        print(f"\nFAIL — {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nPASS — label_schema.yaml is valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
