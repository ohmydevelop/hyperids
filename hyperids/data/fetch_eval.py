"""Fetch public external corpora for out-of-distribution evaluation.

Produces dataset/external_eval/*.jsonl, each line:
  {text, source, label, ...meta}

label semantics:
  malicious   -> clearly offensive commands (GTFOBins / PayloadsAllTheThings)
  benign      -> clearly legitimate commands (NL2Bash)
  attack_session -> real attacker-session commands (Cowrie honeypot); the
                    command text itself may be benign recon, so this is NOT a
                    hard binary label — it is evaluated as a distribution.

No training. Purely for evaluation.
"""
from __future__ import annotations

import json
import re
import subprocess
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "dataset" / "public_raw"
OUT_DIR = ROOT / "dataset" / "external_eval"

COWRIE_URLS = {
    "2021_2022": "https://huggingface.co/datasets/zyw-286/shell-attack-evolution-dataset/resolve/main/commands/2021_2022.jsonl",
    "2024": "https://huggingface.co/datasets/zyw-286/shell-attack-evolution-dataset/resolve/main/commands/2024.jsonl",
}
NL2BASH_URL = "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.cm"
PAYLOADS_URLS = {
    "reverse_shell": "https://raw.githubusercontent.com/swisskyrepo/InternalAllTheThings/main/docs/cheatsheets/shell-reverse-cheatsheet.md",
    "bind_shell": "https://raw.githubusercontent.com/swisskyrepo/InternalAllTheThings/main/docs/cheatsheets/shell-bind-cheatsheet.md",
}

MAX_LEN = 500


def _get(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "hyperids-eval/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


_DESC_LINE = re.compile(r"^(victim|listener|target|attacker|windows|linux|macos|server|client)\s*[:\-]", re.I)


def _extract_code_blocks(md: str) -> list[str]:
    cmds = []
    for block in re.findall(r"```(?:bash|sh|shell|console)?\s*\n(.*?)```", md, re.S):
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # strip leading prompt prefixes
            line = re.sub(r"^[\w@.-]+[\$#>]\s*", "", line).strip()
            if not line or _DESC_LINE.match(line):
                continue
            cmds.append(line)
    return cmds


def fetch_cowrie() -> list[dict]:
    rows = []
    for period, url in COWRIE_URLS.items():
        raw = _get(url)
        for line in raw.splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cmd = (obj.get("command") or "").strip()
            if not cmd or len(cmd) > MAX_LEN:
                continue
            rows.append({"text": cmd, "source": "cowrie", "label": "attack_session",
                         "period": period, "frequency": obj.get("frequency")})
    return rows


def fetch_nl2bash() -> list[dict]:
    raw = _get(NL2BASH_URL)
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or len(line) > MAX_LEN:
            continue
        rows.append({"text": line, "source": "nl2bash", "label": "benign"})
    return rows


def fetch_gtfobins() -> list[dict]:
    repo = RAW / "GTFOBins.github.io"
    if not (repo / "_gtfobins").exists():
        RAW.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/GTFOBins/GTFOBins.github.io.git", str(repo)],
                       check=False, capture_output=True)
    rows = []
    if not (repo / "_gtfobins").exists():
        return rows
    for p in sorted((repo / "_gtfobins").iterdir()):
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
            fm = yaml.safe_load(text)
        except Exception:
            continue
        funcs = fm.get("functions") if isinstance(fm, dict) else None
        if isinstance(funcs, dict):
            for _kind, entries in funcs.items():
                for e in entries or []:
                    if not isinstance(e, dict):
                        continue
                    code = e.get("code")
                    codes = code if isinstance(code, list) else [code]
                    for c in codes:
                        if isinstance(c, str) and c.strip() and len(c.strip()) <= MAX_LEN:
                            rows.append({"text": c.strip(), "source": "gtfobins", "label": "malicious"})
    return rows


def fetch_payloads() -> list[dict]:
    rows = []
    for _kind, url in PAYLOADS_URLS.items():
        try:
            md = _get(url)
        except Exception as e:
            print(f"  skip {url}: {e}")
            continue
        for c in _extract_code_blocks(md):
            if 3 <= len(c) <= MAX_LEN:
                rows.append({"text": c, "source": "payloads", "label": "malicious"})
    return rows


def _dedup(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        k = r["text"]
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = {
        "cowrie": fetch_cowrie,
        "nl2bash": fetch_nl2bash,
        "gtfobins": fetch_gtfobins,
        "payloads": fetch_payloads,
    }
    summary = {}
    for name, fn in sources.items():
        try:
            rows = _dedup(fn())
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
            rows = []
        path = OUT_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[name] = len(rows)
        print(f"[{name}] wrote {len(rows)} -> {path}")
    print("\nSUMMARY:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
