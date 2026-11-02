#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AgroLabel-AL · YOLOv8n-seg 基线训练脚本 (Task-003)
=================================================
用法示例（冒烟，Windows 建议 workers=0）：
  E:\\anaco\\envs\\sam_gpu\\python.exe training/train_seg_baseline.py \
      --epochs 2 --fraction 0.05 --workers 0 --name smoke

正式训练（本地 5060，workers 可设 2~4）：
  python training/train_seg_baseline.py --epochs 150 --workers 4 --name plantseg34_baseline

注意：Windows 多进程安全，必须保留 __main__ 守卫。
"""

import argparse
import os
import sys

# 云端连不上 GitHub / PyPI，ultralytics 启动时的联网自检会卡在 SYN_SENT 十几分钟。
# YOLO_OFFLINE=1 关掉这些检查，纯本地训练不受影响。
os.environ.setdefault("YOLO_OFFLINE", "1")

# 固定工作目录为项目根，保证相对路径 configs/... 可解析
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="configs/dataset_plantseg34.local.yaml")
    p.add_argument("--model", default="yolov8n-seg.pt")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--optimizer", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--close_mosaic", type=int, default=10)
    p.add_argument("--amp", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--cache", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="plantseg34_baseline")
    p.add_argument("--exist_ok", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--val_after", type=lambda x: x.lower() != "false", default=True,
                   help="训练后在 val/test 上评估并输出指标")
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    from ultralytics import YOLO

    print("=" * 60, flush=True)
    print(f"YOLOv8n-seg 基线训练 | data={args.data} epochs={args.epochs} "
          f"fraction={args.fraction} device={args.device} workers={args.workers}", flush=True)
    print("=" * 60, flush=True)

    # 模型文件不存在时 Ultralytics 会去 GitHub 下载，云端常常直接卡死十几分钟。
    # 这里提前判断：给了本地路径但文件不在 → 立刻报错退出，不要静默挂起。
    if os.sep in args.model and not os.path.isfile(args.model):
        raise SystemExit(
            f"模型权重不存在：{args.model}\n"
            f"请把预训练权重放到该路径（云端无法访问 GitHub 自动下载）。"
        )

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        optimizer=args.optimizer,
        seed=args.seed,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        amp=args.amp,
        cache=args.cache,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        verbose=True,
    )
    save_dir = str(model.trainer.save_dir) if model.trainer else None
    print(f"\n[TRAIN DONE] save_dir = {save_dir}", flush=True)

    if args.val_after and save_dir:
        best = os.path.join(save_dir, "weights", "best.pt")
        print(f"\n[VAL] 使用 best.pt 在 val 上评估: {best}", flush=True)
        if os.path.exists(best):
            m2 = YOLO(best)
            metrics = m2.val(data=args.data, split="val", device=args.device, workers=args.workers)
            print("\n==== VAL 指标 ====", flush=True)
            print("box:", metrics.box, flush=True)
            print("seg:", metrics.seg, flush=True)
        else:
            print("[WARN] 未找到 best.pt，跳过 val", flush=True)

    print("\nALL_DONE", flush=True)


if __name__ == "__main__":
    main()
