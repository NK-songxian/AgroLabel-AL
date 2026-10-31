"""PlantDoc 补充评估：跨域鲁棒性 + 真实分辨率下的时延。

## 为什么不能直接算跨域 mAP

两个数据集的标柱粒度不同：PlantSeg 标**病斑**掩码，PlantDoc 标**叶片**外接框。
一个问"叶子上的斑在哪"，一个问"叶子的外框在哪"，类别体系也对不上
（PlantSeg 34 个作物 vs PlantDoc 29 个作物+病害）。硬算 mAP 只会得到一个
既不可解释、也不可复现的数字，不如不算。

## 那就测两件真正能测的事

**一、病斑框的落点合理性（包含率）**
模型在 PlantDoc 上检出的病斑框，应当落在真值叶片框**内部**。
如果模型是瞎猜的，框会散到背景、树枝、天空上，包含率就会掉下来。
这是个不需要类别对齐的跨域信号：只要模型还在"往叶子上找病变"，
包含率就该接近 100%。反过来，若包含率很低，说明跨域迁移失败。

同时报"整图检出率"（有多少张图至少给出一个候选）——这是预标注可用性的下限，
一张图一个候选都没有，标注员就只能从零画。

**二、真实分辨率下的时延**
PlantSeg 的图多是 480×640，而实际业务里手机拍的照片动辄 3000×4000。
比赛要求单张预标注时延 ≤300ms。这里在 PlantDoc 的真实尺寸上测，
比在自己缩过的训练尺寸上测更能说明问题。

用法：
  python src/m9_plantdoc_eval.py --weights runs/plantseg34/weights/best.pt \
      --plantdoc out/dataset_plantdoc --imgsz 640 --n 300
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def load_boxes(label_path, w, h):
    """PlantDoc 的 YOLO 检测标签 -> [(cls, (x1,y1,x2,y2))]，归一化坐标还原成像素。"""
    out = []
    p = Path(label_path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(parts[0])
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        out.append((c, ((cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def containment(inner, outer):
    """inner 框有多少面积落在 outer 框里。"""
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area = max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1e-6)
    return inter / area


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--plantdoc", default="out/dataset_plantdoc")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--n", type=int, default=0, help="最多测多少张；0 = 全部")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--name", default="plantdoc_eval")
    args = ap.parse_args()

    root = Path(args.plantdoc)
    img_dir = root / "images"
    lab_dir = root / "labels"
    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if args.n:
        imgs = imgs[: args.n]
    print(f"PlantDoc {len(imgs)} 张")

    from ultralytics import YOLO
    model = YOLO(args.weights)

    # 预热：第一次前向会带上 CUDA 上下文初始化和 cudnn 选核，必须剔掉
    for p in imgs[: args.warmup]:
        b = imread_unicode(p)
        if b is not None:
            model.predict(b, imgsz=args.imgsz, conf=args.conf,
                          device=args.device, verbose=False)

    lat, n_det, n_img_hit, n_inst = [], 0, 0, 0
    contain_all, contain_cov = [], []
    per_size = []
    crop_hit = {}

    for k, p in enumerate(imgs, 1):
        bgr = imread_unicode(p)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        t0 = time.perf_counter()
        r = model.predict(bgr, imgsz=args.imgsz, conf=args.conf,
                          device=args.device, verbose=False)[0]
        lat.append((time.perf_counter() - t0) * 1000.0)

        leaf = load_boxes(lab_dir / (p.stem + ".txt"), w, h)
        boxes = []
        if r.boxes is not None and len(r.boxes):
            boxes = r.boxes.xyxy.cpu().numpy()
        # 注意 boxes 是 ndarray，不能直接写 `if boxes:`（数组真值有歧义）
        n_inst += len(boxes)
        if len(boxes):
            n_img_hit += 1

        # 每个预测框：落在任一真值叶片框里的最大包含比例
        for b in boxes:
            if not leaf:
                continue
            best = max(containment(b, lb) for _, lb in leaf)
            contain_all.append(best)
            if best >= 0.5:
                contain_cov.append(best)
            # 记一下这个框落在哪一类作物上，做粗粒度的类别覆盖统计
            owner = max(leaf, key=lambda t: containment(b, t[1]))[0]
            crop_hit[owner] = crop_hit.get(owner, 0) + 1

        per_size.append({"image": p.name, "w": w, "h": h,
                         "n_pred": len(boxes), "n_leaf": len(leaf),
                         "ms": round(lat[-1], 1)})

        if k % 200 == 0:
            print(f"  {k}/{len(imgs)}  检出 {n_img_hit} 张  "
                  f"当前均值 {np.mean(lat):.0f}ms")

    n = len(lat)
    res = {
        "weights": args.weights, "n_images": n, "imgsz": args.imgsz,
        "conf": args.conf,
        "image_hit_rate": round(n_img_hit / max(n, 1), 4),
        "mean_pred_per_image": round(n_inst / max(n, 1), 2),
        "latency_ms": {"mean": round(float(np.mean(lat)), 1),
                       "p50": round(pct(lat, 50), 1),
                       "p95": round(pct(lat, 95), 1),
                       "max": round(float(np.max(lat)), 1)},
        "containment": {
            "n_boxes": len(contain_all),
            "mean": round(float(np.mean(contain_all)), 4) if contain_all else None,
            "share_ge_0.5": round(len(contain_cov) / max(len(contain_all), 1), 4)
            if contain_all else None,
            "share_ge_0.9": round(
                sum(1 for v in contain_all if v >= 0.9) / max(len(contain_all), 1), 4)
            if contain_all else None,
        },
        "note": "PlantSeg 标病斑、PlantDoc 标叶片，粒度与类别体系均不可对齐，"
                "因此这里测的是落点合理性与时延，不是精度。",
    }

    out = PROJECT / "out" / "report"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{args.name}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    with open(out / f"{args.name}_per_image.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        import csv
        wtr = csv.DictWriter(f, fieldnames=["image", "w", "h", "n_pred",
                                            "n_leaf", "ms"])
        wtr.writeheader()
        wtr.writerows(per_size)

    print(f"\n=== PlantDoc 跨域检查（{n} 张）===")
    print(f"  整图检出率（至少给出 1 个候选）{res['image_hit_rate']:.1%}")
    print(f"  平均候选数 {res['mean_pred_per_image']}")
    c = res["containment"]
    if c["n_boxes"]:
        print(f"  病斑框落在叶片框内的比例：均值 {c['mean']:.3f}，"
              f"≥50% 覆盖的占 {c['share_ge_0.5']:.1%}，"
              f"≥90% 覆盖的占 {c['share_ge_0.9']:.1%}")
    L = res["latency_ms"]
    print(f"  时延 均值 {L['mean']}ms  P50 {L['p50']}ms  "
          f"P95 {L['p95']}ms  最大 {L['max']}ms")
    print(f"\n结果：{out / f'{args.name}.json'}")


if __name__ == "__main__":
    main()
