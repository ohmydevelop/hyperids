#!/usr/bin/env python3
"""Strip likely-mislabeled action.web_shell from train/val.

Rule (conservative): if the sample has reverse_shell or bind_shell, AND the
command does not look like deploying a file into a webroot / writing php/jsp,
drop action.web_shell. Other labels unchanged. Test is frozen unless --also_test.

Does not execute commands. Does not read private data.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "dataset" / "gliclass_v6"
OUT = ROOT / "dataset" / "gliclass_v7_relabel"

WEB_DEPLOY = re.compile(
    r"(/var/www|/usr/share/nginx|/html/|/uploads/|/wp-content/|"
    r"\.php\b|\.jsp\b|\.aspx\b|\.phtml\b|nginx/html|www/html|"
    r"webshell-detection-lab)",
    re.I,
)
WRITE_PHP = re.compile(
    r"(>\s*\S+\.(php|jsp|aspx|phtml)\b|tee\s+\S+\.(php|jsp|aspx)|"
    r"<\?php|system\(\$_(GET|POST|REQUEST|COOKIE))",
    re.I,
)


def looks_like_web_deploy(text: str) -> bool:
    return bool(WEB_DEPLOY.search(text) or WRITE_PHP.search(text))


def should_drop_web_shell(ex: dict) -> bool:
    labs = set(ex.get("true_labels") or [])
    if "action.web_shell" not in labs:
        return False
    if "action.reverse_shell" not in labs and "action.bind_shell" not in labs:
        return False
    return not looks_like_web_deploy(ex.get("text") or "")


def relabel_split(rows: list[dict]) -> tuple[list[dict], int]:
    out = []
    n = 0
    for ex in rows:
        if should_drop_web_shell(ex):
            n += 1
            new = dict(ex)
            new["true_labels"] = [l for l in ex["true_labels"] if l != "action.web_shell"]
            out.append(new)
        else:
            out.append(ex)
    return out, n


def web_shell_count(rows):
    return sum(1 for e in rows if "action.web_shell" in e.get("true_labels", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=str(SRC))
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--also_test", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in ("train", "val", "test"):
        rows = json.loads((src / f"{split}.json").read_text())
        before = web_shell_count(rows)
        if split == "test" and not args.also_test:
            (out / f"{split}.json").write_text(json.dumps(rows, ensure_ascii=False))
            summary[split] = {"n": len(rows), "web_shell_before": before, "web_shell_after": before, "dropped": 0, "frozen": True}
            print(f"{split}: FROZEN n={len(rows)} web_shell={before}")
            continue
        new_rows, dropped = relabel_split(rows)
        after = web_shell_count(new_rows)
        (out / f"{split}.json").write_text(json.dumps(new_rows, ensure_ascii=False))
        summary[split] = {"n": len(rows), "web_shell_before": before, "web_shell_after": after, "dropped": dropped, "frozen": False}
        print(f"{split}: n={len(rows)} web_shell {before} -> {after} (dropped {dropped})")

    (out / "labels_desc.json").write_text((src / "labels_desc.json").read_text())
    (out / "relabel_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
