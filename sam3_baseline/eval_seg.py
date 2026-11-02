"""评测 SAM3 零样本结果：COCO mask mAP + 逐类 AP + Dice。

输出与 YOLOv8n-seg 的 Ultralytics 指标口径一致（pycocotools segm），
因此 Mask mAP50 / mAP50-95 可直接与 runs/plantseg34_baseline 对比。

用法：
  python -m sam3_baseline.eval_seg --gt runs/sam3_zeroshot/val/gt.json \
      --pred runs/sam3_zeroshot/val/pred.json --out runs/sam3_zeroshot/val
"""
import argparse
import csv
import json
import os

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="")
    return ap.parse_args()


def per_class_ap(ev, coco):
    """从 COCOeval 的 precision 矩阵里取逐类 AP50 与 AP。"""
    p = ev.eval["precision"]  # [T, R, K, A, M]
    k_list = sorted(coco.getCatIds())
    if p.size == 0:
        return []
    rows = []
    for k, cid in enumerate(k_list):
        col50 = p[0, :, k, 0, -1]
        col50 = col50[col50 > -1]
        col = p[:, :, k, 0, -1]
        col = col[col > -1]
        rows.append({
            "class_id": cid - 1,
            "name": coco.loadCats([cid])[0]["name"],
            "ap50": float(col50.mean()) if col50.size else 0.0,
            "ap": float(col.mean()) if col.size else 0.0,
            "n_gt": len(coco.getAnnIds(catIds=[cid])),
        })
    return rows


def _rle(d):
    """json 里的 counts 是 str，pycocotools 需要 bytes。"""
    c = d["counts"]
    if isinstance(c, str):
        c = c.encode("ascii")
    return {"size": list(d["size"]), "counts": c}


def mean_dice(gt_path, pred_path, iou_gate=0.5):
    with open(gt_path) as f:
        gt = json.load(f)
    with open(pred_path) as f:
        preds = json.load(f)
    by_img = {}
    for p in preds:
        by_img.setdefault(p["image_id"], []).append(p)
    dices = []
    for a in gt["annotations"]:
        if a.get("iscrowd", 0):
            continue
        cands = by_img.get(a["image_id"], [])
        if not cands:
            continue
        grle = _rle(a["segmentation"])
        garea = float(mask_utils.area(grle))
        best, best_iou = -1.0, 0.0
        for c in cands:
            prle = _rle(c["segmentation"])
            iou = float(mask_utils.iou([prle], [grle], [0])[0][0])
            if iou > best_iou:
                best_iou = iou
                inter = float(mask_utils.area(mask_utils.merge([prle, grle], True)))
                union = float(mask_utils.area(prle)) + garea - inter
                best = (2 * inter / union) if union > 0 else 0.0
        if best_iou >= iou_gate:
            dices.append(best)
    return float(np.mean(dices)) if dices else 0.0, len(dices)


def main():
    args = parse_args()
    coco_gt = COCO(args.gt)
    with open(args.pred) as f:
        preds = json.load(f)

    res = {"gt": args.gt, "pred": args.pred, "n_pred": len(preds)}
    dice, n_match = mean_dice(args.gt, args.pred)
    res["dice_matched"] = round(dice, 4)
    res["n_matched"] = n_match

    if preds:
        coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, "segm")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        res["mask_mAP50"] = round(float(ev.stats[1]), 4)
        res["mask_mAP50_95"] = round(float(ev.stats[0]), 4)
        res["mask_mAP75"] = round(float(ev.stats[2]), 4)
        rows = per_class_ap(ev, coco_gt)
    else:
        res.update({"mask_mAP50": 0.0, "mask_mAP50_95": 0.0, "mask_mAP75": 0.0})
        rows = []

    print("\n=== SAM3 zero-shot (mask) ===")
    for k in ("mask_mAP50", "mask_mAP75", "mask_mAP50_95", "dice_matched", "n_pred", "n_matched"):
        print(f"{k:>16}: {res[k]}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "sam3_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        if rows:
            with open(os.path.join(args.out, "sam3_per_class.csv"), "w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(sorted(rows, key=lambda r: r["ap50"]))
            print("\n逐类 AP50 后 8 名：")
            for r in sorted(rows, key=lambda r: r["ap50"])[:8]:
                print(f"  {r['name']:<14} AP50={r['ap50']:.3f}  n_gt={r['n_gt']}")


if __name__ == "__main__":
    main()
