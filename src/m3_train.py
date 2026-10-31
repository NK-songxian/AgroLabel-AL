"""M3 第二步：用伪标签训练第一个实例分割模型。

模型选 YOLO11n-seg：
  - n 是最小号，8.5G 显存的笔记本卡能跑，也给后面主动学习留出让模型变大的余地
  - 输入 256×256，正好等于这批图的原生分辨率，不浪费也不糊

注意这一步只是在「验证流程通不通」，不是在刷指标。
伪标签本身有约 10% 是错的，模型会把这部分错误当真理学进去，
所以第一轮 mAP 低是正常的。真正的精度要靠 M4/M5 的人机协同把标签洗一遍。

运行：
  python src/m3_train.py --epochs 40
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from common import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT / "configs" / "default.yaml"))
    ap.add_argument("--data", default=None,
                    help="data.yaml 路径；默认 out/dataset/data.yaml（PlantVillage 伪标签）")
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="seg_pseudo")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", default="disk",
                    choices=["disk", "ram", "none"],
                    help="图像缓存位置。PlantSeg 的原图在机械盘上，"
                         "解一次存成 npy 能让后续 epoch 少掉大部分 IO 等待")
    ap.add_argument("--no-aug", action="store_true",
                    help="关掉强增强；PlantSeg 是自然场景图，翻转过强会引入不合理样本")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="跳过训练，只加载 --weights 做 val/test 评估")
    ap.add_argument("--weights", default=None,
                    help="配合 --eval-only 指定权重")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_yaml = Path(args.data) if args.data else PROJECT / "out" / "dataset" / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"找不到 {data_yaml}，先跑数据集生成脚本")

    from ultralytics import YOLO

    # 增强：PlantVillage 是单叶居中摆拍，翻转安全；
    # PlantSeg 是自然场景（果园、田间、带背景枝叶），上下翻转会造出现实里
    # 不存在的姿态，所以只保留左右翻转和轻度几何扰动。
    aug = dict(fliplr=0.5, flipud=0.0, degrees=10.0, scale=0.5,
               translate=0.1, mosaic=0.0, close_mosaic=0)
    if args.no_aug:
        aug = dict(fliplr=0.0, flipud=0.0, degrees=0.0, scale=0.0,
                   translate=0.0, mosaic=0.0)

    model = YOLO(args.weights if args.eval_only else args.model)
    if args.eval_only:
        print(f"跳过训练，直接评估 {args.weights}")
    else:
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(PROJECT / "runs"),
            name=args.name,
            exist_ok=True,
            resume=args.resume,
            workers=args.workers,
            cache=False if args.cache == "none" else args.cache,
            patience=args.patience,
            val=True,
            plots=True,
            verbose=True,
            **aug,
        )
        # 训练结束后用 best.pt 评估，别用最后一轮的权重
        best_w = PROJECT / "runs" / args.name / "weights" / "best.pt"
        if best_w.exists():
            model = YOLO(str(best_w))
            print(f"评估权重：{best_w}")

    # 把关键指标单独存一份，后面写报告和画主动学习曲线要用
    out = {"model": args.model, "epochs": args.epochs, "imgsz": args.imgsz,
           "data": str(data_yaml)}
    for split in ("val", "test"):
        metrics = model.val(data=str(data_yaml), imgsz=args.imgsz,
                            device=args.device, split=split, verbose=False)
        # 逐类 AP 是细分类别/小样本那 12 分的核心表格，必须存下来
        per_cls = {}
        try:
            names = [metrics.names[i] for i in range(len(metrics.names))]
            per_cls = {n: float(m) for n, m in
                       zip(names, metrics.seg.map50_per_class)}
        except AttributeError:
            try:
                names = [metrics.names[i] for i in range(len(metrics.names))]
                per_cls = {n: float(m) for n, m in zip(names, metrics.box.ap50)}
            except Exception:
                per_cls = {}
        out[split] = {
            "mask_mAP50": float(metrics.seg.map50),
            "mask_mAP50_95": float(metrics.seg.map),
            "box_mAP50": float(metrics.box.map50),
            "box_mAP50_95": float(metrics.box.map),
            "mask_precision": float(metrics.seg.mp),
            "mask_recall": float(metrics.seg.mr),
            "per_class_mask_mAP50": per_cls,
        }
    p = PROJECT / "out" / "report"
    p.mkdir(parents=True, exist_ok=True)
    with open(p / f"{args.name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n=== 结果 ===")
    for split in ("val", "test"):
        m = out[split]
        print(f"  [{split}] 掩码 mAP@0.5 {m['mask_mAP50']:.4f}  "
              f"mAP@0.5:0.95 {m['mask_mAP50_95']:.4f}  "
              f"框 mAP@0.5 {m['box_mAP50']:.4f}")
    # 注意：传了 project 之后 ultralytics 不再自动补任务子目录，
    # 实际目录是 runs/<name>，不是 runs/segment/<name>
    print(f"  权重：{PROJECT / 'runs' / args.name / 'weights' / 'best.pt'}")
    print(f"  指标：{p / f'{args.name}_metrics.json'}")


if __name__ == "__main__":
    main()
