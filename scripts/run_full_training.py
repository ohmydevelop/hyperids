"""上传数据 + 代码 + v4 checkpoint 到 L4，GPU 重训（resume_from v4，save_name v5）。"""
from __future__ import annotations
import os
import time
from pathlib import Path
from lightning_sdk import Studio

STUDIO_NAME = os.environ.get("LIGHTNING_STUDIO_NAME", "")
TEAMSPACE = os.environ.get("LIGHTNING_TEAMSPACE", "")
REMOTE_DIR = "hyperids"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--save_name", type=str, default="final_model_v5")
    ap.add_argument("--resume_from", type=str, default="final_model_v4")
    args = ap.parse_args()

    if not STUDIO_NAME or not TEAMSPACE:
        raise SystemExit("缺少 Lightning 配置：请设置 LIGHTNING_STUDIO_NAME / LIGHTNING_TEAMSPACE（见 .env.example）")
    s = Studio(name=STUDIO_NAME, teamspace=TEAMSPACE, create_ok=False)
    print(f"studio {s.name}: {s.status} on {s.machine}", flush=True)

    print("[1] 上传数据 + 代码 + v4 checkpoint ...", flush=True)
    s.run(f"rm -rf ~/{REMOTE_DIR} && mkdir -p ~/{REMOTE_DIR}/model/checkpoints_gpu")
    s.upload_folder("hyperids", f"{REMOTE_DIR}/hyperids")
    s.upload_folder("dataset/gliclass_v2", f"{REMOTE_DIR}/dataset/gliclass_v2")
    s.upload_folder(f"model/checkpoints_gpu/{args.resume_from}", f"{REMOTE_DIR}/model/checkpoints_gpu/{args.resume_from}")
    time.sleep(20)

    print("[2] 验证文件 ...", flush=True)
    print(s.run(f"ls ~/{REMOTE_DIR}/model/checkpoints_gpu/{args.resume_from}/"), flush=True)

    print(f"[3] GPU 训练 (epochs={args.epochs} batch={args.batch_size} lr={args.lr}) ...", flush=True)
    cmd = (f"cd ~/{REMOTE_DIR} && python -u -m hyperids.train "
           f"--data_dir dataset/gliclass_v2 --save_name {args.save_name} "
           f"--resume_from model/checkpoints_gpu/{args.resume_from} "
           f"--epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr} --device cuda")
    out = s.run(cmd)
    print(out[-5000:], flush=True)

    print(f"[4] 下载 {args.save_name} ...", flush=True)
    local = Path(f"model/checkpoints_gpu/{args.save_name}")
    local.parent.mkdir(parents=True, exist_ok=True)
    s.download_folder(f"{REMOTE_DIR}/model/checkpoints/{args.save_name}", str(local))
    print(f"→ {local}", flush=True)


if __name__ == "__main__":
    main()
