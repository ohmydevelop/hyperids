"""Upload final data to the L4 studio and run full GLiClass training."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from lightning_sdk import Studio

STUDIO_NAME = os.environ.get("LIGHTNING_STUDIO_NAME", "")
TEAMSPACE = os.environ.get("LIGHTNING_TEAMSPACE", "")
REMOTE_DIR = "hyperids"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()

    if not STUDIO_NAME or not TEAMSPACE:
        raise SystemExit("缺少 Lightning 配置：请设置 LIGHTNING_STUDIO_NAME / LIGHTNING_TEAMSPACE（见 .env.example）")
    s = Studio(name=STUDIO_NAME, teamspace=TEAMSPACE, create_ok=False)
    print(f"studio {s.name}: {s.status} on {s.machine}", flush=True)

    print("[1] uploading final data + code ...", flush=True)
    s.run(f"rm -rf ~/{REMOTE_DIR}/hyperids ~/{REMOTE_DIR}/dataset/gliclass_v2 && mkdir -p ~/{REMOTE_DIR}/dataset/gliclass_v2")
    s.upload_folder("dataset/gliclass_v2", f"{REMOTE_DIR}/dataset/gliclass_v2")
    s.upload_folder("hyperids", f"{REMOTE_DIR}/hyperids")
    
    
    
    time.sleep(30)  # let fuse sync
    print("[2] verifying files ...", flush=True)
    print(s.run(f"ls -la ~/{REMOTE_DIR}/dataset/gliclass_v2/"), flush=True)

    print(f"[3] training (epochs={args.epochs} batch={args.batch_size} lr={args.lr}) ...", flush=True)
    out = s.run(
        f"cd ~/{REMOTE_DIR} && python -u -m hyperids.train --data_dir dataset/gliclass_v2 "
        f"--epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr} --device cuda"
    )
    print(out[-5000:], flush=True)
    print("[4] downloading final model ...", flush=True)
    local = Path("model/checkpoints_gpu/final_model")
    local.parent.mkdir(parents=True, exist_ok=True)
    s.download_folder(f"{REMOTE_DIR}/model/checkpoints/final_model", str(local))
    print(f"→ {local}", flush=True)


if __name__ == "__main__":
    main()
