#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8n-seg 基线评估与可视化  (Task-003 交付物生成)
====================================================
功能:
  1) 在 val / test 上评估，输出 Box 与 Mask 的 P/R/mAP50/mAP50-95
  2) 生成混淆矩阵、PR 曲线、val 批图(由 ultralytics 自动绘制)
  3) 在若干 test 图上预测，输出检测框 + 病斑掩码可视化
  4) 汇总指标到 experiments/baseline_metrics.json 与 .md

用法:
  python training/eval_and_visualize.py --weights runs/baseline5/weights/best.pt
Windows 数据加载用 workers=0 以规避多进程问题。
必须保留 __main__ 守卫。
"""

import argparse
import json
import os
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/baseline5/weights/best.pt")
    p.add_argument("--data", default="configs/dataset_plantseg34.local.yaml")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--n_vis", type=int, default=8, help="可视化预测的测试图片数量")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="baseline_eval")
    return p.parse_args()


def metric_dict(m):
    return {
        "precision": float(m.mp),
        "recall": float(m.mr),
        "map50": float(m.map50),
        "map50_95": float(m.map),
        "map75": float(m.map75),
    }


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    from ultralytics import YOLO

    weights = args.weights
    if not os.path.exists(weights):
        raise FileNotFoundError(f"找不到权重: {weights}")

    model = YOLO(weights)
    summary = {"weights": os.path.abspath(weights), "data": args.data, "splits": {}}

    # 1) val / test 评估（自动产出混淆矩阵、PR 曲线、val_batch 可视化于 project/name 下）
    for split in ["val", "test"]:
        print(f"\n==== 评估 split={split} ====", flush=True)
        r = model.val(data=args.data, split=split, device=args.device,
                      workers=args.workers, project=args.project,
                      name=f"{args.name}_{split}", exist_ok=True, plots=True)
        summary["splits"][split] = {
            "box": metric_dict(r.box),
            "seg": metric_dict(r.seg),
            "save_dir": str(r.save_dir),
        }
        print(f"[{split}] Box  mAP50={r.box.map50:.4f} mAP50-95={r.box.map:.4f} | "
              f"Mask mAP50={r.seg.map50:.4f} mAP50-95={r.seg.map:.4f}", flush=True)

    # 2) 检测 + 病斑掩码可视化（对若干 test 图预测并保存）
    test_img_dir = os.path.join("F:/_plantseg/dataset_plantseg_34", "images", "test")
    imgs = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))[: args.n_vis]
    vis_dir = os.path.join("experiments", "baseline_vis")
    os.makedirs(vis_dir, exist_ok=True)
    print(f"\n==== 预测可视化 {len(imgs)} 张 -> {vis_dir} ====", flush=True)
    if imgs:
        model.predict(source=imgs, device=args.device, save=True,
                      project=vis_dir.replace("\\", "/"), name="pred",
                      exist_ok=True, conf=0.25, iou=0.5, workers=0)
    summary["vis_dir"] = os.path.abspath(os.path.join(vis_dir, "pred"))

    # 3) 写出 JSON
    os.makedirs("experiments", exist_ok=True)
    json_path = os.path.join("experiments", "baseline_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] metrics JSON -> {json_path}", flush=True)

    # 4) 写出 Markdown 摘要
    md_path = os.path.join("experiments", "baseline_eval.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# YOLOv8n-seg 基线评估摘要\n\n")
        f.write(f"- 权重: `{summary['weights']}`\n")
        f.write(f"- 数据: `{summary['data']}`\n")
        f.write(f"- 可视化: `{summary['vis_dir']}`\n\n")
        f.write("| split | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for sp, d in summary["splits"].items():
            b, s = d["box"], d["seg"]
            f.write(f"| {sp} | {b['precision']:.4f} | {b['recall']:.4f} | {b['map50']:.4f} | "
                    f"{b['map50_95']:.4f} | {s['precision']:.4f} | {s['recall']:.4f} | "
                    f"{s['map50']:.4f} | {s['map50_95']:.4f} |\n")
        f.write("\n> 注：若为短 epoch 启动基线(v0)，指标偏低属预期；正式 150 epoch 训练后应显著提升。\n")
    print(f"[OK] metrics MD   -> {md_path}", flush=True)
    print("\nEVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
