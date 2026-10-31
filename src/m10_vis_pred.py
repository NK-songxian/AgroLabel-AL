"""把「模型预测」和「人工真值」摆在一起看，一图回答「模型到底行不行」。

## 和 plantseg_check.png 的区别

`plantseg_check.png` 是**转换核对图**：把 PlantSeg 的 PNG 掩码转成多边形后叠回原图，
用来确认坐标没错位（掩码值有 Index+1 偏移，还有 12 张旧类别表的图）。
那张图上没有一个像素来自模型，它证明的是数据对，不是模型好。

这个脚本画的是**预测 vs 真值**：红的是模型现在给出的结果，绿的是人工标注。
第四列是诊断图，把每个实例的匹配情况分色画出来：

  绿 = 匹配上（同类、掩码 IoU ≥ 0.5）—— 标注员只需确认或轻拖
  红 = 漏检（真值有、预测没有）—— 标注员要从零画
  黄 = 误检（预测有、真值没有）—— 标注员要删掉

样例不是随手挑的，按逐图编辑次数从少到多取三档，分别代表
「预标注基本能直接用」「要改一半」「基本没法用」。

用法：
  python src/m10_vis_pred.py --data F:/_plantseg/dataset_plantseg_34/data.yaml \
      --weights runs/plantseg34/weights/best.pt --split test --conf 0.25
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from common import setup_matplotlib  # noqa: E402
from m6_editcost import (IOU_MATCH, iou, load_yolo_seg,  # noqa: E402
                         preds_to_masks, resolve_split, images_to_labels,
                         imread_unicode)

C_GT = (46, 160, 67)        # 真值绿
C_PRED = (200, 60, 55)      # 预测红
C_MISS = (214, 39, 40)      # 漏检红
C_FA = (230, 170, 30)       # 误检黄
C_OK = (46, 160, 67)        # 匹配绿


def match_instances(gt, pred):
    """同类贪心 IoU 匹配，返回 (匹配对, 漏检下标, 误检下标)。"""
    pairs = []
    for i, (gc, gm) in enumerate(gt):
        for j, (pc, pm) in enumerate(pred):
            if gc != pc:
                continue
            v = iou(gm, pm)
            if v >= IOU_MATCH:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_g, used_p, matched = set(), set(), []
    for v, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        matched.append((i, j, v))
    miss = [i for i in range(len(gt)) if i not in used_g]
    fa = [j for j in range(len(pred)) if j not in used_p]
    return matched, miss, fa


def overlay(bgr, masks_colors):
    """在图上叠加半透明掩码 + 轮廓。masks_colors = [(mask, rgb)]。"""
    out = bgr.copy()
    for m, rgb in masks_colors:
        if m.sum() == 0:
            continue
        color = np.array(rgb[::-1], dtype=np.float32)  # rgb -> bgr
        out[m > 0] = (0.55 * out[m > 0] + 0.45 * color).astype(np.uint8)
        cnts, _ = cv2.findContours((m * 255).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, tuple(int(c) for c in color), 2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def pick_samples(per_image_csv, n=3):
    """按编辑次数从少到多分层挑样例。"""
    p = Path(per_image_csv)
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
    if not rows:
        return None
    rows.sort(key=lambda r: int(r["edits"]))
    idx = [0, len(rows) // 2, len(rows) - 1]
    return [rows[i] for i in idx[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--samples", nargs="*", default=None,
                    help="指定图片文件名；不给就按编辑次数自动挑三档")
    ap.add_argument("--per-image", default="out/report/editcost_plantseg34_w_per_image.csv")
    ap.add_argument("--name", default="pred_vs_gt")
    args = ap.parse_args()

    imgs = resolve_split(args.data, args.split)
    labs = images_to_labels(imgs)
    by_name = {p.name: (p, l) for p, l in zip(imgs, labs)}

    picks = pick_samples(args.per_image, 3) if not args.samples else None
    if args.samples:
        names = args.samples
    elif picks:
        names = [r["image"] for r in picks]
    else:
        names = [imgs[i].name for i in range(min(3, len(imgs)))]
    print("样例：", names)

    from ultralytics import YOLO
    model = YOLO(args.weights)

    plt = setup_matplotlib()
    n = len(names)
    fig, axes = plt.subplots(n, 4, figsize=(15.5, 3.9 * n))
    if n == 1:
        axes = np.array([axes])
    cols = ["① 原图", "② 人工真值（绿）", "③ 模型预测（红）", "④ 匹配诊断"]
    for c, t in enumerate(cols):
        axes[0, c].set_title(t, fontsize=13, pad=8)

    for r, name in enumerate(names):
        if name not in by_name:
            print(f"  跳过 {name}（不在 {args.split} 里）")
            for c in range(4):
                axes[r, c].axis("off")
            continue
        ip, lp = by_name[name]
        bgr = imread_unicode(ip)
        h, w = bgr.shape[:2]
        gt = load_yolo_seg(lp, w, h)
        res = model.predict(bgr, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                            device=args.device, verbose=False)[0]
        pred = preds_to_masks(res[0], w, h, args.conf)
        matched, miss, fa = match_instances(gt, pred)

        miou = float(np.mean([v for _, _, v in matched])) if matched else 0.0
        info = (f"{name[:26]}\n真值 {len(gt)} 片 / 预测 {len(pred)} 片\n"
                f"命中 {len(matched)}（掩码IoU {miou:.2f}）/ 漏 {len(miss)} / 误 {len(fa)}")

        axes[r, 0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        axes[r, 0].set_ylabel(info, fontsize=10.5, rotation=0,
                              ha="right", va="center", labelpad=18)

        axes[r, 1].imshow(overlay(bgr, [(m, C_GT) for _, m in gt]))
        axes[r, 2].imshow(overlay(bgr, [(m, C_PRED) for _, m in pred]))

        diag = []
        for i, j, _ in matched:
            diag.append((gt[i][1], C_OK))
        for j in fa:
            diag.append((pred[j][1], C_FA))
        for i in miss:
            diag.append((gt[i][1], C_MISS))
        axes[r, 3].imshow(overlay(bgr, diag))

        for c in range(4):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    def rgb(c):
        # 常量本身就是 RGB，直接归一化给 matplotlib，不要再反转
        return tuple(v / 255.0 for v in c)

    handles = [
        mpatches.Patch(color=rgb(C_OK), label="匹配上（只需确认或轻拖边界）"),
        mpatches.Patch(color=rgb(C_MISS), label="漏检（标注员要从零画）"),
        mpatches.Patch(color=rgb(C_FA), label="误检（标注员要删掉）"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.012))
    plt.suptitle(
        f"PlantSeg {args.split} 集上「模型预测 vs 人工真值」· conf={args.conf}\n"
        f"样例按单张编辑次数从少到多取三档，依次代表「可直接用」「要改一半」「基本没法用」",
        fontsize=13.5)
    plt.tight_layout(rect=[0.02, 0.03, 1, 0.96])

    dst = PROJECT / "out" / "figures"
    dst.mkdir(parents=True, exist_ok=True)
    out_png = dst / f"{args.name}.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {out_png}")


if __name__ == "__main__":
    main()
