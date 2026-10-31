"""M1/M3 通用推理脚本：在图上跑模型，出可视化，顺手量时延。

时延是验收硬指标（单张预标注 ≤300ms），所以不能凭感觉，
每次都要用同一段代码量，把数字记下来。

用法：
  python src/m1_infer.py                                # 抽 8 张测试图，用最新训练权重
  python src/m1_infer.py --n 1 --show                   # 只看一张
  python src/m1_infer.py --weights runs/segment/seg_pseudo/weights/best.pt
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from common import load_config, setup_matplotlib


def imread_unicode(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def default_weights():
    cands = sorted((PROJECT / "runs").rglob("best.pt"))
    return cands[-1] if cands else PROJECT / "yolo11n-seg.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT / "configs" / "default.yaml"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--bench", type=int, default=50, help="计时重复多少次")
    args = ap.parse_args()

    cfg = load_config(args.config)
    weights = Path(args.weights) if args.weights else default_weights()
    if not weights.exists():
        raise SystemExit(f"权重不存在：{weights}")

    # 抽样：从 manifest 里挑
    rows = []
    with open(PROJECT / cfg["paths"]["manifest"], encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["split"] == args.split and r["is_background"] == "0":
                rows.append(r)
    rnd = random.Random(0)
    picks = rnd.sample(rows, min(args.n, len(rows)))

    from ultralytics import YOLO
    model = YOLO(str(weights))

    # 预热：第一次推理包含显存分配和 cudnn 选算法，计进去会虚高
    warm = imread_unicode(picks[0]["path"])
    for _ in range(3):
        model.predict(warm, imgsz=args.imgsz, device=args.device, verbose=False)

    tiles, lat = [], []
    for r in picks:
        bgr = imread_unicode(r["path"])
        if bgr is None:
            continue
        t0 = time.perf_counter()
        res = model.predict(bgr, imgsz=args.imgsz, device=args.device,
                            conf=args.conf, verbose=False)[0]
        lat.append((time.perf_counter() - t0) * 1000)

        ov = bgr.copy()
        n = 0
        if res.masks is not None:
            for poly, cls, conf in zip(res.masks.xy, res.boxes.cls, res.boxes.conf):
                n += 1
                pts = poly.astype(np.int32)
                cv2.polylines(ov, [pts], True, (0, 0, 255), 1)
                x, y = pts[:, 0].min(), pts[:, 1].min()
                cv2.putText(ov, f"{res.names[int(cls)]} {conf:.2f}",
                            (int(x), max(10, int(y) - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1,
                            cv2.LINE_AA)
        tiles.append((ov, n))

    # 计时：单独一段干净循环，只算推理不含画图
    bench_img = imread_unicode(picks[0]["path"])
    t0 = time.perf_counter()
    for _ in range(args.bench):
        model.predict(bench_img, imgsz=args.imgsz, device=args.device,
                      conf=args.conf, verbose=False)
    per = (time.perf_counter() - t0) / args.bench * 1000

    if tiles:
        plt = setup_matplotlib()
        cols = 4
        rows_n = (len(tiles) + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(11, 3 * rows_n))
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (t, n) in zip(axes.ravel(), tiles):
            ax.imshow(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
            ax.set_title(f"检出 {n} 个实例", fontsize=8)
        plt.tight_layout()
        out = PROJECT / cfg["paths"]["figures"] / "infer_grid.png"
        plt.savefig(out, dpi=140)
        plt.close(fig)
        print("可视化：", out)

    print(f"\n权重：{weights.name}")
    print(f"单张预标注时延：{per:.1f} ms（{args.bench} 次均值，imgsz={args.imgsz}）")
    print(f"  首轮 {len(lat)} 张含前后处理：中位 {np.median(lat):.1f} ms")
    print(f"  平均每张检出 {np.mean([n for _, n in tiles]):.2f} 个实例")


if __name__ == "__main__":
    main()
