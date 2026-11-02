#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""F2: YOLOE-Seg 微调（矩阵 F2）

与 train_seg_baseline.py 同构：离线环境、权重存在性检查、训练后 val。
YOLOE 微调会融合类别头（官方行为），训练后提示词不可随意更换，需记录。

用法：
  python training/train_yoloe_seg.py --data data/plantseg34/data.yaml \
      --model weights/yoloe-11l-seg.pt --imgsz 640 --epochs 150 \
      --name f2_yoloe_l_seg_640 --project experiments/matrix/F2/runs
"""
import argparse
import os
import sys

os.environ.setdefault("YOLO_OFFLINE", "1")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="configs/dataset_plantseg34.local.yaml")
    p.add_argument("--model", default="weights/yoloe-11l-seg.pt")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--optimizer", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--close_mosaic", type=int, default=10)
    p.add_argument("--amp", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default="experiments/matrix/F2/runs")
    p.add_argument("--name", default="f2_yoloe_seg")
    p.add_argument("--exist_ok", type=lambda x: x.lower() == "true", default=False)
    return p.parse_args()


def main():
    a = parse_args()
    os.chdir(PROJECT_ROOT)
    from ultralytics import YOLOE

    print(f"[F2] YOLOE-Seg finetune | data={a.data} epochs={a.epochs} "
          f"imgsz={a.imgsz} batch={a.batch} device={a.device}", flush=True)
    if os.sep in a.model and not os.path.isfile(a.model):
        raise SystemExit(f"权重不存在：{a.model}（云端离线，不能自动下载）")

    m = YOLOE(a.model)
    m.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        fraction=a.fraction, optimizer=a.optimizer, seed=a.seed,
        patience=a.patience, close_mosaic=a.close_mosaic, amp=a.amp,
        device=a.device, workers=a.workers, project=a.project, name=a.name,
        exist_ok=a.exist_ok, verbose=True,
    )
    save_dir = str(m.trainer.save_dir) if m.trainer else None
    print(f"\n[TRAIN DONE] save_dir = {save_dir}", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
