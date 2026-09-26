"""Prompt builders for the Frontier LLM pipeline stages.

Prompt strategy (from configs/model_selection.md):
  * response_format=json_object
  * full label list is large (199) — keep instructions tight and rely on
    normalize + validate + retry rather than asking the model to be perfect
  * temperature 0 for labeling, >0 for creative synthesis/red-team
"""
from __future__ import annotations

from schema_v1 import all_label_ids, group_offsets

_OFF = group_offsets()
_ID = all_label_ids()
RISK = list(_ID[_OFF["risk"][0] : _OFF["risk"][1]])
INTENT = list(_ID[_OFF["intent"][0] : _OFF["intent"][1]])
TACTIC = list(_ID[_OFF["tactic"][0] : _OFF["tactic"][1]])
TECH = list(_ID[_OFF["technique"][0] : _OFF["technique"][1]])

_LABEL_BLOCK = (
    "Available risk labels (pick exactly 1): " + ", ".join(RISK) + "\n"
    "Available intent labels (pick any number): " + ", ".join(INTENT) + "\n"
    "Available tactic labels (pick any number): " + ", ".join(TACTIC) + "\n"
    "Available technique labels (MITRE ids, pick any number): " + ", ".join(TECH)
)

_LABEL_SCHEMA_HINT = (
    'Output STRICT JSON only: {"risk": "<one risk label>", '
    '"intents": ["<intent label>", ...], "tactics": ["<tactic label>", ...], '
    '"techniques": ["<technique label>", ...]}. Use ONLY exact labels from the lists.'
)


def label_prompt(command: str) -> list[dict]:
    system = (
        "You are a security command/script classifier for a multi-label "
        "classification model. Assign risk, intents, MITRE tactics, and MITRE "
        "techniques to the command. Choose labels ONLY from the provided lists. "
        "If none applies for a list, use an empty array."
    )
    user = f"{_LABEL_BLOCK}\n\n{_LABEL_SCHEMA_HINT}\n\nCommand to classify:\n{command}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def synthesize_prompt(labels: list[str], n: int = 1) -> list[dict]:
    system = (
        "You generate realistic shell commands / short scripts for a security "
        "training dataset. Generate plausible, varied samples consistent with "
        "the requested labels. Output STRICT JSON only: {\"samples\": [\"cmd1\", \"cmd2\"]}."
    )
    user = (
        f"Labels: {', '.join(labels)}\n"
        f"Generate {n} distinct shell command(s)/script(s) that exactly match "
        f"these labels (risk + intent + tactic + technique). Be concrete and realistic."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def obfuscate_prompt(command: str, labels: list[str]) -> list[dict]:
    system = (
        "You create obfuscated variants of a shell command that keep the same "
        "security behavior. Use encoding, indirection, case changes, env vars, "
        "quoting, etc. Output STRICT JSON only: {\"variants\": [\"cmd1\", ...]}."
    )
    user = (
        f"Original command:\n{command}\n"
        f"Its labels: {', '.join(labels)}\n"
        f"Produce 3 obfuscated variants that preserve the same behavior and labels."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def hard_negative_prompt(command: str, labels: list[str]) -> list[dict]:
    system = (
        "You generate near-miss 'hard negative' shell commands that look "
        "malicious but are actually benign (or vice versa), to make a classifier "
        "more robust. Output STRICT JSON only: {\"samples\": [{\"command\": \"...\", \"labels\": [\"...\"]}]}."
    )
    user = (
        f"Reference command:\n{command}\nReference labels: {', '.join(labels)}\n"
        f"Generate 3 near-miss commands that superficially resemble this but have "
        f"a DIFFERENT risk (benign<->malicious) or different intent. "
        f"Use ONLY labels from these lists:\n{_LABEL_BLOCK}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def minimal_pair_prompt(command: str, labels: list[str]) -> list[dict]:
    system = (
        "You create minimal pairs: two shell commands that differ by a single "
        "small edit but flip benign<->malicious. Output STRICT JSON only: "
        "{\"pairs\": [{\"benign\": \"...\", \"malicious\": \"...\"}]}."
    )
    user = (
        f"Seed command:\n{command}\nSeed labels: {', '.join(labels)}\n"
        f"Create 2 minimal pairs (one small edit changes the risk). "
        f"Use ONLY labels from these lists:\n{_LABEL_BLOCK}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def diverse_prompt(n: int) -> list[dict]:
    """Bulk diverse command generation (unlabeled — Jev is the labeler)."""
    system = (
        "You generate a large, diverse corpus of realistic shell commands and "
        "short scripts for training a security classifier. Vary shells, OS "
        "flavors, tools, quoting styles, and complexity. Output STRICT JSON only: "
        '{"commands": ["cmd1", "cmd2", ...]}.'
    )
    user = (
        f"Generate {n} distinct, realistic shell commands/scripts. "
        "Balance roughly: ~40% benign (sysadmin, devops, monitoring, builds, "
        "package install, docker/k8s, git, file/network ops, backups); "
        "~30% malicious (reverse shell, download-and-execute, persistence, "
        "credential access, defense evasion, discovery, exfiltration, impact, "
        "C2, lateral movement); ~30% suspicious/dual-use (recon, SUID checks, "
        "log clearing, history tampering, encoders, scanners). "
        "Make each command distinct — vary targets, paths, ports, IPs, tools. "
        "No explanations, just the JSON array."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def benign_prompt(n: int) -> list[dict]:
    system = (
        "You generate a large corpus of realistic, legitimate shell commands and "
        "short scripts (sysadmin / devops / monitoring / builds / package mgmt / "
        "docker / k8s / git / file & network ops / backups / logs). "
        'Output STRICT JSON only: {"commands": ["cmd1", "cmd2", ...]}.'
    )
    user = (
        f"Generate {n} distinct, realistic BENIGN shell commands/scripts. "
        "All must be clearly legitimate with no security impact. Vary tools, "
        "paths, flags, OS flavors, quoting styles. No explanations."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def suspicious_prompt(n: int) -> list[dict]:
    system = (
        "You generate dual-use / suspicious shell commands for a security "
        "training dataset: recon, SUID checks, history tampering, log clearing, "
        "base64/hex encoding, port scanning, credential probing, process/network "
        "inspection. They should look ambiguous but not clearly destructive. "
        'Output STRICT JSON only: {"commands": ["cmd1", "cmd2", ...]}.'
    )
    user = (
        f"Generate {n} distinct SUSPICIOUS/dual-use shell commands. Vary tools, "
        "targets, flags. No explanations."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
