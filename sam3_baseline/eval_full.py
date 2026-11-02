"""统一评测协议：六项指标（S0–S4 共用同一套口径）

  Mask AP50 / AP50-95     —— pycocotools segm，逐类平均
  Instance Recall         —— IoU≥0.5 贪心匹配的 TP/(TP+FN)，maxDets=100
  Instance Coverage       —— 至少有一个预测匹配上的 GT 实例占比
  Matched Dice            —— 已匹配实例的 Dice 均值（=2IoU/(1+IoU)）
  Boundary F1             —— 已匹配实例的边界 F1（2px 容差）

用法：
  python -m sam3_baseline.eval_full --gt runs/.../gt.json --pred runs/.../pred.json --out runs/...
"""
import argparse
import csv
import json
import os

import numpy as np
from PIL import Image, ImageFilter
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

IOU_GATE = 0.5
BOUNDARY_TOL_PX = 2


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--boundary-max", type=int, default=3000, help="边界 F1 最多算多少个匹配对")
    return ap.parse_args()


def _rle(d):
    c = d["counts"]
    if isinstance(c, str):
        c = c.encode("ascii")
    return {"size": list(d["size"]), "counts": c}


def _to_img(mask_bool):
    return Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")


def boundary_f1(mp, mg):
    """2px 容差下的边界 F1。"""
    a, b = _to_img(mp), _to_img(mg)
    bp = np.asarray(a).astype(np.int16) - np.asarray(a.filter(ImageFilter.MinFilter(3))).astype(np.int16) > 0
    bg = np.asarray(b).astype(np.int16) - np.asarray(b.filter(ImageFilter.MinFilter(3))).astype(np.int16) > 0
    if not bp.any() and not bg.any():
        return 1.0
    dp = np.asarray(_to_img(bp).filter(ImageFilter.MaxFilter(2 * BOUNDARY_TOL_PX + 1))) > 0
    dg = np.asarray(_to_img(bg).filter(ImageFilter.MaxFilter(2 * BOUNDARY_TOL_PX + 1))) > 0
    p = float((bp & dg).sum()) / max(1, int(bp.sum()))
    r = float((bg & dp).sum()) / max(1, int(bg.sum()))
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def per_class_ap(ev, coco):
    p = ev.eval["precision"]
    rows = []
    if p.size == 0:
        return rows
    for k, cid in enumerate(sorted(coco.getCatIds())):
        c50 = p[0, :, k, 0, -1]
        c50 = c50[c50 > -1]
        allc = p[:, :, k, 0, -1]
        allc = allc[allc > -1]
        rows.append({"class_id": cid - 1, "name": coco.loadCats([cid])[0]["name"],
                     "ap50": round(float(c50.mean()) if c50.size else 0.0, 4),
                     "ap": round(float(allc.mean()) if allc.size else 0.0, 4),
                     "n_gt": len(coco.getAnnIds(catIds=[cid]))})
    return rows


def compute_metrics(gt, preds, boundary_max=3000, gt_path=None):
    """gt/preds 均为 COCO 结构；纯内存计算，便于批量扫描。"""
    import tempfile

    res = {"n_pred": len(preds)}
    rows = []
    tmp = None
    if preds:
        if gt_path is None:
            tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
            json.dump(gt, tmp)
            tmp.close()
            gt_path = tmp.name
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, "segm")
        ev.evaluate()
        ev.accumulate()
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            ev.summarize()  # stats 只有 summarize 后才填充
        res.update({"mask_mAP50": round(float(ev.stats[1]), 4),
                    "mask_mAP75": round(float(ev.stats[2]), 4),
                    "mask_mAP50_95": round(float(ev.stats[0]), 4),
                    "coco_AR100": round(float(ev.stats[8]), 4)})
        rows = per_class_ap(ev, coco_gt)
    else:
        res.update({"mask_mAP50": 0.0, "mask_mAP75": 0.0, "mask_mAP50_95": 0.0, "coco_AR100": 0.0})

    by_img = {}
    for p in preds:
        by_img.setdefault(p["image_id"], []).append(p)
    tp = n_gt = n_cov = 0
    dices, bf1s = [], []
    for ann in gt["annotations"]:
        n_gt += 1
        cands = by_img.get(ann["image_id"])
        if not cands:
            continue
        g = _rle(ann["segmentation"])
        best_iou, best_p = 0.0, None
        for c in cands:
            pr = _rle(c["segmentation"])
            iou = float(mask_utils.iou([pr], [g], [0])[0][0])
            if iou > best_iou:
                best_iou, best_p = iou, pr
        if best_iou >= IOU_GATE:
            tp += 1
            n_cov += 1
            dices.append(2.0 * best_iou / (1.0 + best_iou))
            if len(bf1s) < boundary_max:
                bf1s.append(boundary_f1(mask_utils.decode(best_p) > 0, mask_utils.decode(g) > 0))

    res.update({"instance_recall": round(tp / max(1, n_gt), 4),
                "instance_coverage": round(n_cov / max(1, n_gt), 4),
                "matched_dice": round(float(np.mean(dices)), 4) if dices else 0.0,
                "boundary_f1": round(float(np.mean(bf1s)), 4) if bf1s else 0.0,
                "n_gt": n_gt, "n_matched": tp})
    if tmp is not None:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return res, rows


def main():
    a = parse_args()
    gt = json.load(open(a.gt, encoding="utf-8"))
    preds = json.load(open(a.pred, encoding="utf-8"))
    res, rows = compute_metrics(gt, preds, a.boundary_max, gt_path=a.gt)
    res.update({"gt": a.gt, "pred": a.pred})
    print("\n=== 统一评测（六项） ===")
    for k in ("mask_mAP50", "mask_mAP50_95", "mask_mAP75", "instance_recall",
              "instance_coverage", "matched_dice", "boundary_f1"):
        print(f"{k:>18}: {res[k]}")
    print(f"{'n_pred / n_matched':>18}: {res['n_pred']} / {res['n_matched']}")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump(res, open(os.path.join(a.out, "metrics_full.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        if rows:
            with open(os.path.join(a.out, "per_class.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(sorted(rows, key=lambda r: -r["ap50"]))


if __name__ == "__main__":
    main()
