"""Orchestrate GLiClass fine-tuning on the Lightning L4 studio via lightning-sdk."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from lightning_sdk import Studio

STUDIO_NAME = "<YOUR_STUDIO>"
TEAMSPACE = "<YOUR_LIGHTNING_ACCOUNT>/<YOUR_TEAMSPACE>"
REMOTE_DIR = "hyperids"

SETUP_CMD = (
    "pip install -q --root-user-action=ignore gliclass==0.1.20 'transformers==4.57.6' "
    "accelerate protobuf sentencepiece tiktoken scikit-learn pyyaml"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--setup_only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip upload/setup, just train")
    args = ap.parse_args()

    s = Studio(name=STUDIO_NAME, teamspace=TEAMSPACE, create_ok=False)
    print(f"studio: {s.name}  status={s.status}  machine={s.machine}", flush=True)

    if not args.resume:
        print("[1/3] installing deps on studio ...", flush=True)
        out = s.run(f"cd ~ && {SETUP_CMD}")
        print(out[-1500:], flush=True)

        print("[2/3] uploading code + data ...", flush=True)
        s.run(f"rm -rf ~/{REMOTE_DIR} && mkdir -p ~/{REMOTE_DIR}/hyperids ~/{REMOTE_DIR}/dataset")
        s.upload_folder("hyperids", f"{REMOTE_DIR}/hyperids")
        s.upload_folder("dataset/gliclass_v2", f"{REMOTE_DIR}/dataset/gliclass_v2")
        
        
        print("upload done", flush=True)

    if args.setup_only:
        return

    print(f"[3/3] training (epochs={args.epochs}, batch={args.batch_size}) ...", flush=True)
    cmd = (
        f"cd ~/{REMOTE_DIR} && python -u -m hyperids.train "
        f"--epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr} --device cuda"
    )
    out = s.run(cmd)
    print(out[-4000:], flush=True)
    print("training finished", flush=True)

    # download final model
    local = Path("model/checkpoints_gpu")
    local.mkdir(parents=True, exist_ok=True)
    print("downloading final model ...", flush=True)
    s.download_folder(f"{REMOTE_DIR}/model/checkpoints/final_model", str(local))
    print(f"→ {local}", flush=True)


if __name__ == "__main__":
    main()
