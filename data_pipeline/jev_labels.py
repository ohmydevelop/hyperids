"""Map the HyperIDs 199-label schema onto Jev SystemOne questions.

Decomposition (one request, evaluated in parallel):
    risk      -> 1 Choice question  (mutually exclusive -> softmax probabilities)
    intent    -> 41 Noul questions
    tactic    -> 14 Noul questions
    technique -> 141 Noul questions
    total     -> 197 questions -> one 199-dim probability vector per command

The 199-dim vector order is exactly schema_v1.all_label_ids() order
(risk 3, intent 41, tactic 14, technique 141), so it plugs straight into
Student distillation as soft targets.
"""
from __future__ import annotations

from functools import lru_cache

from data_pipeline import jev_client
from schema_v1 import all_label_ids, group_offsets, label_descriptions

_OFF = group_offsets()
_IDS = all_label_ids()
_DESC = label_descriptions()

_RISK_IDS = list(_IDS[_OFF["risk"][0] : _OFF["risk"][1]])
_INTENT_IDS = list(_IDS[_OFF["intent"][0] : _OFF["intent"][1]])
_TACTIC_IDS = list(_IDS[_OFF["tactic"][0] : _OFF["tactic"][1]])
_TECH_IDS = list(_IDS[_OFF["technique"][0] : _OFF["technique"][1]])

RISK_KEY = "__risk__"


@lru_cache(maxsize=1)
def build_questions() -> dict:
    """Build the cached Jev question dict from label_schema_v1.yaml."""
    q: dict = {
        RISK_KEY: {
            "type": "choice",
            "instructions": "Classify the security risk level of this shell command/script.",
            "criteria": {rid: _DESC[rid] for rid in _RISK_IDS},
        }
    }
    for lid in _INTENT_IDS:
        q[lid] = {"type": "noul", "instructions": f"The command/script has this attacker intent: {_DESC[lid]}"}
    for lid in _TACTIC_IDS:
        q[lid] = {"type": "noul", "instructions": f"The command/script maps to this MITRE tactic: {_DESC[lid]}"}
    for lid in _TECH_IDS:
        q[lid] = {"type": "noul", "instructions": f"The command/script uses this MITRE technique: {_DESC[lid]}"}
    return q


def classify(state: str, model: str = jev_client.DEFAULT_MODEL) -> jev_client.JevResult:
    """Run the full 199-label classification on one command."""
    return jev_client.systemone(state, build_questions(), model=model)


def soft_vector(result: jev_client.JevResult) -> list[float]:
    """199-dim soft target vector in schema order from a JevResult."""
    answers = result.answers
    vec: list[float] = []
    # risk: choice probabilities, default 0.0 for missing options
    risk_ans = answers.get(RISK_KEY) or {}
    probs = risk_ans.get("probabilities") or {}
    vec.extend(float(probs.get(rid, 0.0)) for rid in _RISK_IDS)
    # intent / tactic / technique: noul probabilities
    for lid in list(_INTENT_IDS) + list(_TACTIC_IDS) + list(_TECH_IDS):
        a = answers.get(lid) or {}
        vec.append(float(a.get("noul", 0.0)))
    return vec


def hard_labels(result: jev_client.JevResult, threshold: float = 0.5) -> list[str]:
    """Binarize a JevResult into canonical label ids (risk argmax + noul>threshold)."""
    answers = result.answers
    risk_ans = answers.get(RISK_KEY) or {}
    probs = risk_ans.get("probabilities") or {}
    risk = max(_RISK_IDS, key=lambda rid: float(probs.get(rid, 0.0))) if probs else None
    out: list[str] = []
    if risk:
        out.append(risk)
    for lid in list(_INTENT_IDS) + list(_TACTIC_IDS) + list(_TECH_IDS):
        a = answers.get(lid) or {}
        if float(a.get("noul", 0.0)) >= threshold:
            out.append(lid)
    return out
