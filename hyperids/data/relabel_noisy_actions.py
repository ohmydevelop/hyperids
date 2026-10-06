#!/usr/bin/env python3
"""Strip noisy extra action labels that are actually reverse/bind shells.

- action.web_shell: drop if reverse/bind present and command is not webroot deploy
- action.keylog: drop if reverse/bind present and command has no keylog evidence
Other labels unchanged. Applies to train/val/test when --also_test.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "dataset" / "gliclass_v6"
OUT = ROOT / "dataset" / "gliclass_v12_unified"

WEB_DEPLOY = re.compile(
    r"(/var/www|/usr/share/nginx|/html/|/uploads/|/wp-content/|"
    r"\.php\b|\.jsp\b|\.aspx\b|\.phtml\b|nginx/html|www/html|webshell-detection-lab)", re.I)
WRITE_PHP = re.compile(
    r"(>\s*\S+\.(php|jsp|aspx|phtml)\b|tee\s+\S+\.(php|jsp|aspx)|<\?php|system\(\$_(GET|POST|REQUEST|COOKIE))", re.I)
KEYLOG = re.compile(
    r"(keylog|evtest|/dev/input|logkeys|xinput|script\s+-q|strace|keystroke|kbd)", re.I)


def is_web_deploy(text: str) -> bool:
    return bool(WEB_DEPLOY.search(text) or WRITE_PHP.search(text))


def relabel(ex: dict) -> tuple[dict, list[str]]:
    labs = list(ex.get("true_labels") or [])
    text = ex.get("text") or ""
    dropped = []
    has_rb = ("action.reverse_shell" in labs) or ("action.bind_shell" in labs)
    if has_rb and "action.web_shell" in labs and not is_web_deploy(text):
        labs = [l for l in labs if l != "action.web_shell"]
        dropped.append("action.web_shell")
    if has_rb and "action.keylog" in labs and not KEYLOG.search(text):
        labs = [l for l in labs if l != "action.keylog"]
        dropped.append("action.keylog")
    if not dropped:
        return ex, []
    new = dict(ex)
    new["true_labels"] = labs
    return new, dropped


def count(rows, label):
    return sum(1 for e in rows if label in e.get("true_labels", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--also_test", action="store_true")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "val", "test"):
        rows = json.loads((src / f"{split}.json").read_text())
        before = {k: count(rows, k) for k in ("action.web_shell", "action.keylog")}
        if split == "test" and not args.also_test:
            (out / f"{split}.json").write_text(json.dumps(rows, ensure_ascii=False))
            summary[split] = {"n": len(rows), "before": before, "after": before, "dropped": {}, "frozen": True}
            print(f"{split}: FROZEN n={len(rows)} ws={before['action.web_shell']} kl={before['action.keylog']}")
            continue
        new_rows, dropped_c = [], {}
        for ex in rows:
            nxt, dropped = relabel(ex)
            new_rows.append(nxt)
            for d in dropped:
                dropped_c[d] = dropped_c.get(d, 0) + 1
        after = {k: count(new_rows, k) for k in ("action.web_shell", "action.keylog")}
        (out / f"{split}.json").write_text(json.dumps(new_rows, ensure_ascii=False))
        summary[split] = {"n": len(rows), "before": before, "after": after, "dropped": dropped_c, "frozen": False}
        print(f"{split}: n={len(rows)} ws {before['action.web_shell']}->{after['action.web_shell']} "
              f"kl {before['action.keylog']}->{after['action.keylog']} dropped={dropped_c}")
    (out / "labels_desc.json").write_text((src / "labels_desc.json").read_text())
    (out / "relabel_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
