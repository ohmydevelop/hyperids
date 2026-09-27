"""Jev labeling under the v2 schema (31 labels: verdict 3 + action 28).

Reduced from the old 199-dim prompt to 29 questions:
    1 Choice (verdict, 3 options) + 28 Noul (action)
-> one 31-dim soft vector aligned to schema.all_label_ids() order.
"""
from __future__ import annotations

from functools import lru_cache

from data_pipeline import jev_client
import schema

IDS = schema.all_label_ids()
OFF = schema.group_offsets()
VERDICT_IDS = list(IDS[OFF["verdict"][0] : OFF["verdict"][1]])
ACTION_IDS = list(IDS[OFF["action"][0] : OFF["action"][1]])
DESC = schema.descriptions()

VERDICT_KEY = "__verdict__"


@lru_cache(maxsize=1)
def build_questions() -> dict:
    q = {
        VERDICT_KEY: {
            "type": "choice",
            "instructions": "Classify the security risk level of this shell command/script.",
            "criteria": {vid: DESC[vid] for vid in VERDICT_IDS},
        }
    }
    for aid in ACTION_IDS:
        q[aid] = {"type": "noul", "instructions": f"The command/script performs this action: {DESC[aid]}"}
    return q


def classify(state: str, model: str = jev_client.DEFAULT_MODEL) -> jev_client.JevResult:
    return jev_client.systemone(state, build_questions(), model=model)


def soft_vector(result: jev_client.JevResult) -> list[float]:
    """31-dim soft vector in schema.all_label_ids() order."""
    answers = result.answers
    probs = (answers.get(VERDICT_KEY) or {}).get("probabilities") or {}
    vec = [float(probs.get(vid, 0.0)) for vid in VERDICT_IDS]
    for aid in ACTION_IDS:
        a = answers.get(aid) or {}
        vec.append(float(a.get("noul", 0.0)))
    return vec


def hard_labels(result: jev_client.JevResult, threshold: float = 0.5) -> list[str]:
    vec = soft_vector(result)
    verdict = VERDICT_IDS[max(range(3), key=lambda i: vec[i])]
    actions = [aid for i, aid in enumerate(ACTION_IDS) if vec[3 + i] >= threshold]
    return [verdict] + actions
