# -*- coding: utf-8 -*-
"""
i03_ondemand_refine.py —— 轻量预标注 + 按需 SAM3 精修（矩阵 I-03）

三组对照（同一批图、同一 YOLO 权重、同一 SAM3 权重）：
  none     轻量模型（YOLO-seg）直接出预标注，不做任何精修        —— 延迟下限
  full     每个实例的框都送 SAM3 重算掩码                        —— 精度上限 / 延迟上限
  ondemand 只把"可疑实例"的框送 SAM3 精修（本脚本的核心策略）     —— 折中点

"可疑"的判定必须可解释、可调、不依赖真值（真值只在评测时用）：
  R1 置信度低      conf < --tau-conf
  R2 掩码不完整    掩码面积 / 框面积 < --min-fill
  R3 掩码碎裂      连通域数 > 1
  R4 目标太小      掩码面积 / 图面积 < --min-area-ratio（小目标边界最不稳）
任一命中即精修。

为什么"精修"用框提示而不是文本提示
--------------------------------
精修的对象是**已经存在的预标注实例**，类别已知、位置已知，唯一不确定的是
边界像素。框提示把"在哪"这个信息直接喂给 SAM3，比让它自己用文本重新找一遍
更省也更准（F4 已量化：GT 框 vs YOLO 框 vs 文本提示三条路）。

指标
----
  mask_mAP50 / mAP50-95 / matched_dice / boundary_f1   —— 质量（sam3_baseline.eval_full 口径）
  精修调用率 = 精修实例数 / 实例总数
  单图延迟 P50 / P95（毫秒）—— 含 YOLO 前向与 SAM3 编码，按组分别统计
  人工秒数 = 编辑单位 × --sec-per-unit（编辑单位沿用 src/m6_editcost.py 的 realistic 权重）

用法
----
  python src/i03_ondemand_refine.py --yolo-weights weights/baseline_yolov8nseg_plantseg34_best.pt \
      --ckpt weights/sam3.pt --data-root data/plantseg34 --split val \
      --out runtime/logs/matrix/I-03 --limit 100
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import ssl
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context
os.environ.setdefault("YOLO_OFFLINE", "1")

from pycocotools import mask as mask_utils  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from m6_editcost import WEIGHT_PRESETS, edit_ops  # noqa: E402
from sam3_baseline.eval_full import compute_metrics  # noqa: E402
from sam3_baseline.zeroshot_infer import (load_json, polys_to_rle,  # noqa: E402
                                          read_label, rle_json)


def parse_args():
    ap = argparse.ArgumentParser(description="轻量预标注 + 按需 SAM3 精修（I-03）")
    ap.add_argument("--yolo-weights", required=True)
    ap.add_argument("--ckpt", required=True, help="SAM3 权重")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--yolo-imgsz", type=int, default=640)
    ap.add_argument("--imgsz", type=int, default=1008, help="SAM3 输入尺寸")
    ap.add_argument("--conf", type=float, default=0.001, help="YOLO 预标注阈值（对齐编辑成本口径）")
    ap.add_argument("--iou", type=float, default=0.9, help="SAM3 内部 NMS IoU")
    ap.add_argument("--tau-conf", type=float, default=0.5, help="R1 置信度阈值")
    ap.add_argument("--min-fill", type=float, default=0.30, help="R2 掩码/框面积比下限")
    ap.add_argument("--min-area-ratio", type=float, default=0.005, help="R4 掩码占图面积比下限")
    ap.add_argument("--max-boxes", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--boundary-max", type=int, default=300)
    ap.add_argument("--sec-per-unit", type=float, default=30.0,
                    help="1 个编辑单位 ≈ 多少秒（画一片病斑约 30s，按实际标注测速可改）")
    ap.add_argument("--weight-preset", default="realistic", choices=list(WEIGHT_PRESETS))
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def poly_mask(poly, W, H):
    import cv2

    m = np.zeros((H, W), np.uint8)
    xy = np.clip(poly * np.array([W, H], dtype=np.float32), 0, None)
    if len(xy) >= 3:
        cv2.fillPoly(m, [np.round(xy).astype(np.int32)], 1)
    return m


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = ua + ub - inter
    return inter / u if u > 0 else 0.0


def need_refine(mask, box, conf, W, H, a, n_comp):
    """三条可解释规则，命中任一即精修。返回 (是否需要, 命中原因列表)。"""
    why = []
    if conf < a.tau_conf:
        why.append("低置信")
    bw, bh = max(1e-6, box[2] - box[0]), max(1e-6, box[3] - box[1])
    area = float(mask.sum())
    if area / (bw * bh) < a.min_fill:
        why.append("掩码不完整")
    if n_comp > 1:
        why.append("掩码碎裂")
    if area / float(W * H) < a.min_area_ratio:
        why.append("目标过小")
    return (len(why) > 0), why


def pct(v, q):
    return float(np.percentile(v, q)) if v else None


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)

    from ultralytics import YOLO
    from ultralytics.models.sam import SAM3SemanticPredictor
    import cv2

    hosts = load_json(os.path.join(a.data_root, "classes.json"))
    hosts = hosts if isinstance(hosts, list) else hosts["classes"]

    img_dir = os.path.join(a.data_root, "images", a.split)
    lbl_dir = os.path.join(a.data_root, "labels", a.split)
    files = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if a.limit:
        files = files[: a.limit]

    yolo = YOLO(a.yolo_weights)
    sam = SAM3SemanticPredictor(overrides=dict(
        task="segment", mode="predict", model=a.ckpt, imgsz=a.imgsz,
        retina_masks=True, verbose=False, save=False, conf=0.0, iou=a.iou))

    gt = {"images": [], "annotations": [],
          "categories": [{"id": i + 1, "name": h} for i, h in enumerate(hosts)]}
    preds = {k: [] for k in ("none", "full", "ondemand")}
    # 每组的：编辑单位、延迟、实例数、精修实例数
    stat = {k: {"units": 0.0, "ms": [], "n_inst": 0, "n_refine": 0} for k in preds}
    why_counter = Counter()
    rows_csv = []
    ann_id = 1

    for i, fn in enumerate(files):
        stem = os.path.splitext(fn)[0]
        with Image.open(os.path.join(img_dir, fn)) as im:
            im = im.convert("RGB")
            W, H = im.size
            arr = np.asarray(im)
            img_id = i + 1
            gt["images"].append({"id": img_id, "file_name": fn, "width": W, "height": H})

            # ---- GT（COCO 用 + 编辑成本用）----
            ids, polys = read_label(os.path.join(lbl_dir, stem + ".txt"))
            gt_masks = []
            for cid, poly in zip(ids, polys):
                r = polys_to_rle([poly], W, H)
                gt["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": cid + 1,
                    "segmentation": rle_json(r), "area": float(mask_utils.area(r)),
                    "bbox": mask_utils.toBbox(r).tolist(), "iscrowd": 0})
                ann_id += 1
                gt_masks.append((int(cid), poly_mask(poly, W, H)))

            # ---- 一次 YOLO 预标注 ----
            t0 = time.time()
            r = yolo.predict(source=arr, imgsz=a.yolo_imgsz, conf=a.conf,
                             device=a.device, verbose=False)[0]
            t_yolo = (time.time() - t0) * 1000.0

            insts = []  # (cls, mask, box, conf)
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                order = np.argsort(-cf)[: a.max_boxes]
                mm_all = r.masks.data.cpu().numpy() if r.masks is not None else None
                for k in order:
                    m = (mm_all[k] > 0.5).astype(np.uint8) if mm_all is not None else None
                    if m is None:
                        continue
                    if m.shape != (H, W):
                        m = (np.asarray(Image.fromarray(m * 255).resize((W, H), Image.NEAREST)) > 127).astype(np.uint8)
                    if m.sum() == 0:
                        continue
                    insts.append((int(cls[k]), m, [float(v) for v in xyxy[k]], float(cf[k])))

            # ---- 三组 ----
            # SAM3 精修：一次编码，两次提示（full 全量 / ondemand 子集）
            m_full = m_ondemand = None
            needs = []
            idx_od = []
            if insts:
                for (c, m, b, conf) in insts:
                    ncomp = cv2.connectedComponents(m.astype(np.uint8))[0] - 1
                    ok, why = need_refine(m, b, conf, W, H, a, ncomp)
                    needs.append(ok)
                    if ok:
                        why_counter.update(why)

                t0 = time.time()
                sam.set_image(arr)
                t_enc = (time.time() - t0) * 1000.0

                def refine(boxes_idx):
                    if not boxes_idx:
                        return None
                    b_in = np.asarray([insts[k][2] for k in boxes_idx], dtype=np.float32)
                    t = time.time()
                    try:
                        ms, ob = sam.inference_features(
                            sam.features, (H, W), bboxes=b_in,
                            labels=np.ones(len(b_in), dtype=np.int32))
                    except Exception as e:  # noqa: BLE001
                        print(f"[warn] {fn} 精修失败: {type(e).__name__}: {e}", flush=True)
                        return None
                    cost = (time.time() - t) * 1000.0
                    if ms is None or ms.shape[0] == 0:
                        return None
                    ms = ms.cpu().numpy()
                    ob = ob.cpu().numpy()
                    out = {}
                    for m2, row in zip(ms, ob):
                        best_k, best_v = -1, 0.0
                        for j, k in enumerate(boxes_idx):
                            v = box_iou(row[:4].tolist(), insts[k][2])
                            if v > best_v:
                                best_v, best_k = v, j
                        if best_k < 0 or best_v < 0.3:
                            continue
                        mm2 = np.ascontiguousarray(m2.astype(np.uint8))
                        if mm2.shape != (H, W):
                            mm2 = (np.asarray(Image.fromarray(mm2 * 255).resize((W, H), Image.NEAREST)) > 127).astype(np.uint8)
                        if mm2.sum() == 0:
                            continue
                        out[boxes_idx[best_k]] = mm2
                    return out, cost

                m_full_r = refine(list(range(len(insts))))
                idx_od = [k for k, ok in enumerate(needs) if ok]
                m_od_r = refine(idx_od)
                m_full = ("full", m_full_r[0] if m_full_r else {}, (m_full_r[1] if m_full_r else 0.0))
                m_ondemand = ("ondemand", m_od_r[0] if m_od_r else {}, (m_od_r[1] if m_od_r else 0.0))
                enc_ms = t_enc
            else:
                enc_ms = 0.0

            for gname in ("none", "full", "ondemand"):
                ms_list = []
                for k, (c, m, b, conf) in enumerate(insts):
                    used = m
                    if gname == "full" and m_full:
                        used = m_full[1].get(k, m)
                    elif gname == "ondemand" and m_ondemand:
                        used = m_ondemand[1].get(k, m)
                    ms_list.append((c, used))

                # 延迟：none 只算 YOLO；full/ondemand 加上编码和各自的提示耗时
                if gname == "none" or not insts:
                    latency = t_yolo
                else:
                    latency = t_yolo + enc_ms + (m_full[2] if gname == "full" else m_ondemand[2])
                    stat[gname]["n_refine"] += (len(insts) if gname == "full" else len(idx_od))

                stat[gname]["ms"].append(latency)
                stat[gname]["n_inst"] += len(insts)
                n_ops, ops, _ = edit_ops(gt_masks, ms_list)
                w = WEIGHT_PRESETS[a.weight_preset]
                stat[gname]["units"] += sum(w.get(k, 1.0) * v for k, v in ops.items())

                for c, m in ms_list:
                    rle = mask_utils.encode(np.asfortranarray(m))
                    preds[gname].append({
                        "image_id": img_id, "category_id": int(c) + 1,
                        "segmentation": rle_json(rle),
                        "score": 1.0,
                        "bbox": [float(v) for v in mask_utils.toBbox(rle)],
                    })

            if insts:
                rows_csv.append({
                    "image": fn, "n_inst": len(insts),
                    "n_refine": sum(1 for x in needs if x),
                    "yolo_ms": round(t_yolo, 1), "enc_ms": round(enc_ms, 1),
                })

        if (i + 1) % 20 == 0 or i + 1 == len(files):
            print(f"[{i+1}/{len(files)}] none_inst={stat['none']['n_inst']} "
                  f"refine_rate={stat['ondemand']['n_refine']/max(1,stat['ondemand']['n_inst']):.3f}",
                  flush=True)

    json.dump(gt, open(os.path.join(a.out, "gt.json"), "w", encoding="utf-8"))
    table = []
    for gname, plist in preds.items():
        json.dump(plist, open(os.path.join(a.out, f"pred_{gname}.json"), "w", encoding="utf-8"))
        m, _ = compute_metrics(gt, plist, a.boundary_max)
        s = stat[gname]
        n = max(1, s["n_inst"])
        row = {
            "group": gname,
            "mask_mAP50": round(float(m["mask_mAP50"]), 4),
            "mask_mAP50_95": round(float(m["mask_mAP50_95"]), 4),
            "mask_mAP75": round(float(m["mask_mAP75"]), 4),
            "matched_dice": round(float(m["matched_dice"]), 4),
            "boundary_f1": round(float(m["boundary_f1"]), 4),
            "instance_recall": round(float(m["instance_recall"]), 4),
            "n_inst": s["n_inst"],
            "refine_rate": round(s["n_refine"] / n, 4),
            "latency_p50_ms": round(pct(s["ms"], 50) or 0.0, 1),
            "latency_p95_ms": round(pct(s["ms"], 95) or 0.0, 1),
            "edit_units": round(s["units"], 2),
            "human_sec": round(s["units"] * a.sec_per_unit, 1),
        }
        table.append(row)
        print(f"{gname:<9} mAP50={row['mask_mAP50']:.4f} dice={row['matched_dice']:.4f} "
              f"P95={row['latency_p95_ms']:.0f}ms refine={row['refine_rate']:.3f} "
              f"human={row['human_sec']:.0f}s", flush=True)

    rep = {
        "experiment": "I-03",
        "split": a.split, "images": len(files),
        "yolo_weights": a.yolo_weights, "ckpt": a.ckpt,
        "rules": {"tau_conf": a.tau_conf, "min_fill": a.min_fill,
                  "min_area_ratio": a.min_area_ratio},
        "weight_preset": a.weight_preset, "sec_per_unit": a.sec_per_unit,
        "table": table,
        "refine_reasons": dict(why_counter),
    }
    json.dump(rep, open(os.path.join(a.out, "i03_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(os.path.join(a.out, "per_image.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows_csv[0]) if rows_csv else ["image"])
        wr.writeheader()
        wr.writerows(rows_csv)

    base = next((r for r in table if r["group"] == "none"), None)
    od = next((r for r in table if r["group"] == "ondemand"), None)
    if base and od and base["human_sec"] > 0:
        print(f"\n按需精修 vs 不精修：人工秒数 {base['human_sec']:.0f}s -> {od['human_sec']:.0f}s "
              f"（{(od['human_sec'] - base['human_sec']) / base['human_sec'] * 100:+.1f}%），"
              f"mAP50 {base['mask_mAP50']:.4f} -> {od['mask_mAP50']:.4f}")
    print(f"DONE -> {a.out}")


if __name__ == "__main__":
    main()
