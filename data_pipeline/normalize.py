"""Normalize raw LLM classification output into canonical schema labels.

The gateway models do not reliably emit canonical ids (they produce
"malicious", "suspicious", free-text intent names, MITRE ids without the
"technique." prefix, etc.). This module maps any such surface form back to
exact ids in label_schema_v1.yaml and rejects anything unmappable.

All matching is deterministic and schema-driven — no extra LLM calls.
"""
from __future__ import annotations

import re
from functools import lru_cache

from schema_v1 import all_label_ids, label_descriptions

_RISK_ALIASES = {
    "benign": "risk.benign",
    "safe": "risk.benign",
    "clean": "risk.benign",
    "normal": "risk.benign",
    "suspicious": "risk.suspicious",
    "suspect": "risk.suspicious",
    "ambiguous": "risk.suspicious",
    "malicious": "risk.malicious",
    "malware": "risk.malicious",
    "dangerous": "risk.malicious",
    "harmful": "risk.malicious",
    "evil": "risk.malicious",
    "high": "risk.malicious",
    "low": "risk.benign",
    "medium": "risk.suspicious",
}

# intent.* 的常见别名：模型常直接输出短语而不是 id
_INTENT_ALIASES = {
    "download": "intent.download",
    "download_and_execute": "intent.download_execute",
    "download_execute": "intent.download_execute",
    "download and execute": "intent.download_execute",
    "dropper": "intent.download_execute",
    "execute": "intent.execute",
    "execution": "intent.execute",
    "run": "intent.execute",
    "persistence": "intent.persistence",
    "persist": "intent.persistence",
    "scheduled_task": "intent.scheduled_task",
    "scheduled task": "intent.scheduled_task",
    "cron": "intent.scheduled_task",
    "credential_access": "intent.credential_access",
    "credential access": "intent.credential_access",
    "credential_dumping": "intent.credential_access",
    "reverse_shell": "intent.reverse_shell",
    "reverse shell": "intent.reverse_shell",
    "bind_shell": "intent.bind_shell",
    "bind shell": "intent.bind_shell",
    "web_shell": "intent.web_shell",
    "web shell": "intent.web_shell",
    "keylog": "intent.keylog",
    "keylogger": "intent.keylog",
    "ransomware": "intent.ransomware",
    "cryptomining": "intent.cryptomining",
    "mining": "intent.cryptomining",
    "data_staging": "intent.data_staging",
    "disable_security": "intent.disable_security",
    "disable security": "intent.disable_security",
    "clear_logs": "intent.clear_logs",
    "clear logs": "intent.clear_logs",
    "timestomp": "intent.timestomp",
    "add_account": "intent.add_account",
    "modify_registry": "intent.modify_registry",
    "service_creation": "intent.service_creation",
    "self_propagation": "intent.self_propagation",
    "worm": "intent.self_propagation",
    "network_scan": "intent.network_scan",
    "scan": "intent.network_scan",
    "reconnaissance": "intent.reconnaissance",
    "recon": "intent.reconnaissance",
    "system_info": "intent.system_info",
    "system info": "intent.system_info",
    "file_transfer": "intent.file_transfer",
    "environment_setup": "intent.environment_setup",
    "initial_access": "intent.initial_access",
    "privilege_escalation": "intent.privilege_escalation",
    "priv esc": "intent.privilege_escalation",
    "defense_evasion": "intent.defense_evasion",
    "discovery": "intent.discovery",
    "lateral_movement": "intent.lateral_movement",
    "collection": "intent.collection",
    "command_and_control": "intent.command_and_control",
    "c2": "intent.command_and_control",
    "c&c": "intent.command_and_control",
    "exfiltration": "intent.exfiltration",
    "exfil": "intent.exfiltration",
    "impact": "intent.impact",
    "destruction": "intent.impact",
    "brute_force": "intent.brute_force",
    "bruteforce": "intent.brute_force",
    "process_inject": "intent.process_inject",
    "file_operation": "intent.file_operation",
    "probe_security_tools": "intent.probe_security_tools",
    "backdoor": "intent.backdoor",
}


@lru_cache(maxsize=1)
def _id_set() -> set[str]:
    return set(all_label_ids())


@lru_cache(maxsize=1)
def _technique_ids() -> set[str]:
    return {x for x in _id_set() if x.startswith("technique.")}


def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9.&\-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def map_risk(raw: str) -> str | None:
    """Map a risk surface form to risk.* or None."""
    n = _norm(raw).replace("risk.", "")
    if n in _RISK_ALIASES:
        return _RISK_ALIASES[n]
    return None


def map_intent(raw: str) -> str | None:
    n = _norm(raw)
    if n.startswith("intent."):
        return n if n in _id_set() else None
    if n in _INTENT_ALIASES:
        return _INTENT_ALIASES[n]
    # try exact id match against group
    cand = f"intent.{n}"
    if cand in _id_set():
        return cand
    return None


def map_tactic(raw: str) -> str | None:
    n = _norm(raw)
    if n.startswith("tactic."):
        return n if n in _id_set() else None
    cand = f"tactic.{n}"
    return cand if cand in _id_set() else None


def map_technique(raw: str) -> str | None:
    """Map MITRE id surface forms to technique.Txxxx or None.

    Accepts: "T1059", "t1059", "technique.T1059", "MITRE T1059", "T1059.001".
    Schema ids use an uppercase "T" ("technique.T1059.004"), so we reconstruct
    that exact form regardless of input casing.
    """
    n = _norm(raw)
    body = n.removeprefix("technique.")
    m = re.search(r"t(\d{4}(?:\.\d{3})?)", body)
    if not m:
        return None
    cand = f"technique.T{m.group(1)}"
    return cand if cand in _id_set() else None


def normalize_risk(raw) -> str | None:
    return map_risk(raw)


def normalize_multi(items, mapper) -> list[str]:
    """Map a list of raw values; drop None; preserve order; dedupe."""
    out: list[str] = []
    seen: set[str] = set()
    for x in items or []:
        got = mapper(x)
        if got and got not in seen:
            seen.add(got)
            out.append(got)
    return out


def validate_labels(labels: list[str]) -> bool:
    """True iff every label is a canonical schema id."""
    ids = _id_set()
    return all(x in ids for x in labels)


def extract_labels(parsed: dict) -> dict[str, list[str]]:
    """From a parsed model JSON, return {group: canonical labels}, with
    best-effort key detection (risk/risk_level/intents/intent/techniques/...).
    """
    # risk
    risk_raw = parsed.get("risk") or parsed.get("risk_level") or parsed.get("classification")
    risk = [normalize_risk(risk_raw)] if risk_raw is not None else []
    risk = [x for x in risk if x]

    intents_raw = parsed.get("intents") or parsed.get("intent") or []
    if isinstance(intents_raw, str):
        intents_raw = [intents_raw]
    intents = normalize_multi(intents_raw, map_intent)

    tactics_raw = parsed.get("tactics") or parsed.get("tactic") or []
    if isinstance(tactics_raw, str):
        tactics_raw = [tactics_raw]
    tactics = normalize_multi(tactics_raw, map_tactic)

    techs_raw = parsed.get("techniques") or parsed.get("technique") or parsed.get("mitre_techniques") or []
    if isinstance(techs_raw, str):
        techs_raw = [techs_raw]
    techniques = normalize_multi(techs_raw, map_technique)

    return {"risk": risk, "intent": intents, "tactic": tactics, "technique": techniques}


def combined_labels(parsed: dict) -> list[str]:
    d = extract_labels(parsed)
    # order must follow schema group order
    return d["risk"] + d["intent"] + d["tactic"] + d["technique"]
