# -*- coding: utf-8 -*-
"""
m13_box_prompt_eval.py —— 框提示 SAM3 评测（矩阵 F4）

矩阵定义：**YOLO 框 + SAM3 / GT 框 + SAM3**，用同一条 SAM3 权重，
只换"框从哪来"，从而把"检测器质量"和"分割器质量"拆开：
  - GT 框   → 分割器上限（oracle detection）
  - YOLO 框 → 端到端实际水平（检测误差会传导到掩码）

为什么要单独一个脚本
--------------------
`sam3_baseline/prompt_sweep.py` 只走**文本提示**（`predictor(text=[...])`），
拿不到"框提示"这条路径；而 SAM3 的框提示在 ultralytics 里的入口是

    SAM3SemanticPredictor.inference_features(features, src_shape, bboxes=xyxy_px, labels=...)

（见 ultralytics/models/sam/predict.py:2342，`_prepare_geometric_prompts` 把
xyxy 像素框转成归一化 xywh 后交给 grounding head）。这条路径**不吃类别文本**，
`nc` 恒为 1（内部自动塞一个 "visual" 提示），所以类别必须由我们自己从框来源
带过来——这也是本脚本比文本提示版本多出来的那段"掩码↔框配对"逻辑。

一个重要细节：`inference_features` 返回的掩码顺序**不可假定**与输入框一致
（内部会做 NMS 且类别位相同、框被重排）。所以下面按"输出掩码框 vs 输入框
最大 IoU"来配对类别，而不是按下标。

用法
----
  # GT 框（上限对照）
  python src/m13_box_prompt_eval.py --ckpt weights/sam3.pt \
      --data-root data/plantseg34 --split val --box-src gt \
      --out runtime/logs/matrix/F4/gt_box --limit 100

  # YOLO 框（端到端）
  python src/m13_box_prompt_eval.py --ckpt weights/sam3.pt \
      --data-root data/plantseg34 --split val --box-src yolo \
      --yolo-weights weights/baseline_yolov8nseg_plantseg34_best.pt \
      --out runtime/logs/matrix/F4/yolo_box --limit 100
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import time

import numpy as np
from PIL import Image

# 离线兜底：云端连不上 GitHub，ultralytics 任何联网自检都会卡死
ssl._create_default_https_context = ssl._create_unverified_context
os.environ.setdefault("YOLO_OFFLINE", "1")

from pycocotools import mask as mask_utils  # noqa: E402

from sam3_baseline.eval_full import compute_metrics  # noqa: E402
from sam3_baseline.zeroshot_infer import (load_json, polys_to_rle,  # noqa: E402
                                          read_label, rle_json)


def parse_args():
    ap = argparse.ArgumentParser(description="框提示 SAM3 掩码评测（F4）")
    ap.add_argument("--ckpt", required=True, help="SAM3 权重")
    ap.add_argument("--data-root", required=True, help="plantseg34 目录（含 images/ labels/ classes.json）")
    ap.add_argument("--split", default="val")
    ap.add_argument("--box-src", choices=["gt", "yolo"], default="gt")
    ap.add_argument("--yolo-weights", default="", help="--box-src yolo 时的检测/分割权重")
    ap.add_argument("--yolo-imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001, help="YOLO 框阈值（宁低勿高，AP 靠排序）")
    ap.add_argument("--iou", type=float, default=0.9, help="SAM3 内部 NMS 的 IoU（病斑密集，放宽）")
    ap.add_argument("--imgsz", type=int, default=1008)
    ap.add_argument("--max-boxes", type=int, default=40, help="每图最多送多少框（控制显存）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张，冒烟用")
    ap.add_argument("--boundary-max", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action="store_true", help="quantize=16，省显存")
    return ap.parse_args()


def box_iou(a, b):
    """单框 IoU（xyxy）。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = ua + ub - inter
    return inter / u if u > 0 else 0.0


def gt_boxes(lbl_path, W, H):
    """从 YOLO-seg 标签还原 xyxy 框，类别沿用标签里的 cls。"""
    ids, polys = read_label(lbl_path)
    out = []
    for cid, poly in zip(ids, polys):
        xy = poly * np.array([W, H], dtype=np.float32)
        if len(xy) < 3:
            continue
        x1, y1 = float(xy[:, 0].min()), float(xy[:, 1].min())
        x2, y2 = float(xy[:, 0].max()), float(xy[:, 1].max())
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append((cid, [x1, y1, x2, y2]))
    return out


def build_predictor(ckpt, imgsz, iou, half):
    from ultralytics.models.sam import SAM3SemanticPredictor

    ov = dict(task="segment", mode="predict", model=ckpt, imgsz=imgsz,
              retina_masks=True, verbose=False, save=False,
              conf=0.0, iou=iou)
    if half:
        ov["quantize"] = 16
    return SAM3SemanticPredictor(overrides=ov)


def build_yolo(weights, imgsz, device):
    from ultralytics import YOLO

    return YOLO(weights)


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)

    hosts = load_json(os.path.join(a.data_root, "classes.json"))
    hosts = hosts if isinstance(hosts, list) else hosts["classes"]

    img_dir = os.path.join(a.data_root, "images", a.split)
    lbl_dir = os.path.join(a.data_root, "labels", a.split)
    files = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if a.limit:
        files = files[: a.limit]

    pred = build_predictor(a.ckpt, a.imgsz, a.iou, a.half)
    yolo = build_yolo(a.yolo_weights, a.yolo_imgsz, a.device) if a.box_src == "yolo" else None

    gt = {"images": [], "annotations": [],
          "categories": [{"id": i + 1, "name": h} for i, h in enumerate(hosts)]}
    preds = []
    ann_id = 1
    t_enc, t_prompt = [], []
    n_box_in = n_box_out = n_pair_fail = 0

    for i, fn in enumerate(files):
        stem = os.path.splitext(fn)[0]
        with Image.open(os.path.join(img_dir, fn)) as im:
            im = im.convert("RGB")
            W, H = im.size
            img_id = i + 1
            gt["images"].append({"id": img_id, "file_name": fn, "width": W, "height": H})

            # ---- GT ----
            ids, polys = read_label(os.path.join(lbl_dir, stem + ".txt"))
            for cid, poly in zip(ids, polys):
                r = polys_to_rle([poly], W, H)
                gt["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": cid + 1,
                    "segmentation": rle_json(r), "area": float(mask_utils.area(r)),
                    "bbox": mask_utils.toBbox(r).tolist(), "iscrowd": 0})
                ann_id += 1

            # ---- 框来源 ----
            if a.box_src == "gt":
                boxes = gt_boxes(os.path.join(lbl_dir, stem + ".txt"), W, H)
            else:
                r = yolo.predict(source=np.asarray(im), imgsz=a.yolo_imgsz,
                                 conf=a.conf, device=a.device, verbose=False)[0]
                boxes = []
                if r.boxes is not None and len(r.boxes):
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    cls = r.boxes.cls.cpu().numpy().astype(int)
                    cf = r.boxes.conf.cpu().numpy()
                    order = np.argsort(-cf)
                    for k in order:
                        boxes.append((int(cls[k]), [float(v) for v in xyxy[k]]))
            if len(boxes) > a.max_boxes:
                boxes = boxes[: a.max_boxes]
            if not boxes:
                continue

            # ---- SAM3 框提示 ----
            b_in = np.asarray([b for _, b in boxes], dtype=np.float32)
            t0 = time.time()
            pred.set_image(np.asarray(im))
            t_enc.append(time.time() - t0)

            t0 = time.time()
            try:
                masks, out_boxes = pred.inference_features(
                    pred.features, (H, W), bboxes=b_in,
                    labels=np.ones(len(b_in), dtype=np.int32))
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {fn} 框提示失败: {type(e).__name__}: {e}", flush=True)
                continue
            t_prompt.append(time.time() - t0)

            if masks is None or masks.shape[0] == 0:
                continue
            masks = masks.cpu().numpy()
            ob = out_boxes.cpu().numpy()

            n_box_in += len(b_in)
            # 输出掩码 ↔ 输入框按 IoU 配对（NMS 后顺序不可信）
            for m, ob_row in zip(masks, ob):
                o_b = ob_row[:4].tolist()
                best_k, best_v = -1, 0.0
                for k, (_, ib) in enumerate(boxes):
                    v = box_iou(o_b, ib)
                    if v > best_v:
                        best_v, best_k = v, k
                if best_k < 0 or best_v < 0.3:
                    n_pair_fail += 1
                    continue
                cid = boxes[best_k][0]
                mm = np.ascontiguousarray(m.astype(np.uint8))
                if mm.shape != (H, W):
                    mm = (np.asarray(Image.fromarray(mm * 255).resize((W, H), Image.NEAREST)) > 127).astype(np.uint8)
                if mm.sum() == 0:
                    continue
                rle = mask_utils.encode(np.asfortranarray(mm))
                bx = mask_utils.toBbox(rle).tolist()
                preds.append({
                    "image_id": img_id, "category_id": int(cid) + 1,
                    "segmentation": rle_json(rle),
                    "score": float(ob_row[4]),
                    "bbox": [float(v) for v in bx],
                })
                n_box_out += 1

        if (i + 1) % 20 == 0 or i + 1 == len(files):
            print(f"[{i+1}/{len(files)}] preds={len(preds)} "
                  f"enc={np.mean(t_enc):.3f}s prompt={np.mean(t_prompt) if t_prompt else 0:.3f}s",
                  flush=True)

    json.dump(gt, open(os.path.join(a.out, "gt.json"), "w", encoding="utf-8"))
    json.dump(preds, open(os.path.join(a.out, "pred.json"), "w", encoding="utf-8"))

    m, _ = compute_metrics(gt, preds, a.boundary_max)
    rep = {
        "experiment": "F4",
        "box_src": a.box_src,
        "yolo_weights": a.yolo_weights if a.box_src == "yolo" else "",
        "ckpt": a.ckpt,
        "split": a.split,
        "imgsz": a.imgsz,
        "conf": a.conf,
        "images": len(files),
        "n_box_in": n_box_in,
        "n_box_out": n_box_out,
        "n_pair_fail": n_pair_fail,
        "enc_sec_mean": round(float(np.mean(t_enc)), 4) if t_enc else None,
        "prompt_sec_mean": round(float(np.mean(t_prompt)), 4) if t_prompt else None,
        "metrics": {k: m[k] for k in ("mask_mAP50", "mask_mAP50_95", "mask_mAP75",
                                      "instance_recall", "instance_coverage",
                                      "matched_dice", "boundary_f1",
                                      "n_pred", "n_matched")},
    }
    json.dump(rep, open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"\n=== F4 框提示（{a.box_src} 框）===")
    for k, v in rep["metrics"].items():
        print(f"{k:>18}: {v}")
    print(f"{'enc_sec_mean':>18}: {rep['enc_sec_mean']}")
    print(f"{'prompt_sec_mean':>18}: {rep['prompt_sec_mean']}")
    print(f"DONE -> {a.out}")


if __name__ == "__main__":
    main()
