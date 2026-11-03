# -*- coding: utf-8 -*-
"""m15_sam3_prompt_sweep.py —— SAM3 提示词工程扫描（Z1s）

## 目的
与 m14（YOLO-World/YOLOE 的 7 模板族扫描，结论：提示对该类模型完全无效）形成对称对照，
验证"提示词工程只对语义分割模型（SAM3）有效"这一假设。

## 方法
复用 m14 的 TEMPLATES（7 个模板族）与 host_prompt_bank 生成 34 条/族的短语；
每个模板族 = 34 条短语，对应 34 个类别（索引 = class_id）。
对每张图只做一次 set_image（缓存图像编码），随后对 7 个族各调用一次
`predictor(text=[34短语])`，SAM3 内部 set_classes 后返回 34 类的掩码；
按 boxes 第 5 列（类别索引）分桶到 34 类，统一用 sam3_baseline.eval_full.compute_metrics
计算 mask mAP50 / mAP50-95 / matched_dice。

## 关键事实（来自 ultralytics SAM3SemanticPredictor.postprocess）
  pred_cls = arange(nc)，NMS 按类别偏移；返回的 Results.boxes 第 5 列即 text 列表索引（0..33）。
  因此 pred.category_id = cls + 1，与 gt（category_id = class_id + 1）严格对齐。

## 成本
模型只加载一次；每图 set_image 一次，7 族各一次 text 推理。1008 尺寸、846 图约 30–60 min。
冒烟用 --limit 60。

## 用法
  python src/m15_sam3_prompt_sweep.py --data-root data/plantseg34 --ckpt weights/sam3.pt \
      --host-bank configs/sam3/host_prompt_bank.json --split val \
      --out experiments/matrix/Z1s
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from pycocotools import mask as mask_utils  # noqa: E402

from m14_prompt_sweep import TEMPLATES, build_prompts  # noqa: E402
from sam3_baseline.eval_full import compute_metrics  # noqa: E402
from sam3_baseline.zeroshot_infer import load_json, polys_to_rle, rle_json, read_label  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="SAM3 提示词工程扫描")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--host-bank", default="configs/sam3/host_prompt_bank.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=1008)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-masks", type=int, default=120,
                    help="每图每族最多保留的实例数（按置信截断）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张，冒烟用")
    ap.add_argument("--families", default=",".join(TEMPLATES),
                    help=f"逗号分隔，可选：{','.join(TEMPLATES)}")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)

    from ultralytics.models.sam import SAM3SemanticPredictor

    classes = load_json(os.path.join(a.data_root, "classes.json"))
    classes = classes if isinstance(classes, list) else classes["classes"]
    n_c = len(classes)

    def _name(c):
        return c if isinstance(c, str) else c.get("name", str(c))

    gt = {"images": [], "annotations": [],
          "categories": [{"id": i + 1, "name": _name(c)}
                         for i, c in enumerate(classes)]}

    fams = [f.strip() for f in a.families.split(",") if f.strip()]
    bad = [f for f in fams if f not in TEMPLATES]
    if bad:
        raise SystemExit(f"未知模板族 {bad}，可选 {list(TEMPLATES)}")

    # 预生成每族 34 条短语（索引 = class_id）；names 仅用长度占位
    names_holder = [f"c{i}" for i in range(n_c)]
    fam_prompts = {f: build_prompts(a.host_bank, names_holder, f) for f in fams}

    predictor = SAM3SemanticPredictor(overrides=dict(
        conf=a.conf, task="segment", mode="predict", model=a.ckpt,
        imgsz=a.imgsz, retina_masks=True, quantize=16, verbose=False, save=False,
        device=a.device))

    img_dir = os.path.join(a.data_root, "images", a.split)
    lbl_dir = os.path.join(a.data_root, "labels", a.split)
    files = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if a.limit:
        files = files[: a.limit]

    preds = {f: [] for f in fams}
    ann_id, t_all, n_img = 1, 0.0, 0
    for i, fn in enumerate(files):
        stem = os.path.splitext(fn)[0]
        with Image.open(os.path.join(img_dir, fn)) as im:
            im = im.convert("RGB")
            W, H = im.size
            img_id = i + 1
            gt["images"].append({"id": img_id, "file_name": fn, "width": W, "height": H})
            ids, polys = read_label(os.path.join(lbl_dir, stem + ".txt"))
            for cid, poly in zip(ids, polys):
                r = polys_to_rle([poly], W, H)
                gt["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": int(cid) + 1,
                    "segmentation": rle_json(r), "area": float(mask_utils.area(r)),
                    "bbox": mask_utils.toBbox(r).tolist(), "iscrowd": 0})
                ann_id += 1
            if not ids:
                continue
            t0 = time.time()
            predictor.set_image(np.asarray(im))
            t_all += time.time() - t0
            n_img += 1
            for f in fams:
                phrases = fam_prompts[f]
                try:
                    out = predictor(text=phrases)[0]
                except Exception:  # noqa: BLE001
                    continue
                if i < 3 and f == fams[0]:
                    nm = None if out.masks is None else len(out.masks)
                    nb = None if out.boxes is None else len(out.boxes)
                    cr = "n/a" if out.boxes is None else \
                        f"{float(out.boxes.cls.min()):.0f}..{float(out.boxes.cls.max()):.0f}"
                    print(f"  [dbg] img{i} {f}: masks={nm} boxes={nb} cls={cr}", flush=True)
                if out.masks is None or len(out.masks) == 0:
                    continue
                ms = out.masks.data.cpu().numpy()
                bx = out.boxes.xyxy.cpu().numpy().reshape(-1, 4)
                cf = out.boxes.conf.cpu().numpy().reshape(-1)
                cl = out.boxes.cls.cpu().numpy().reshape(-1).astype(int)
                if len(cf) > a.max_masks:
                    keep = np.argsort(-cf)[: a.max_masks]
                    ms, bx, cf, cl = ms[keep], bx[keep], cf[keep], cl[keep]
                for m, b, s, c in zip(ms, bx, cf, cl):
                    if float(s) < a.conf:
                        continue
                    if c < 0 or c >= n_c:
                        continue
                    mm = m.astype(np.uint8)
                    if mm.shape != (H, W):
                        mm = (np.asarray(Image.fromarray(mm * 255)
                                         .resize((W, H), Image.NEAREST)) > 127).astype(np.uint8)
                    preds[f].append({
                        "image_id": img_id, "category_id": int(c) + 1,
                        "segmentation": rle_json(mask_utils.encode(np.asfortranarray(mm))),
                        "score": float(s),
                        "bbox": [float(b[0]), float(b[1]),
                                 float(b[2] - b[0]), float(b[3] - b[1])]})
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            print(f"[{i + 1}/{len(files)}] enc_avg={t_all / max(1, n_img):.3f}s "
                  f"preds={{{', '.join(f'{f}:{len(preds[f])}' for f in fams)}}}", flush=True)

    json.dump(gt, open(os.path.join(a.out, "gt.json"), "w", encoding="utf-8"))
    table = []
    for f in fams:
        json.dump(preds[f], open(os.path.join(a.out, f"pred_{f}.json"), "w", encoding="utf-8"))
        m, _ = compute_metrics(gt, preds[f], 300)
        row = {"family": f, "prompt_example": fam_prompts[f][0],
               "mask_mAP50": round(float(m["mask_mAP50"]), 5),
               "mask_mAP50_95": round(float(m["mask_mAP50_95"]), 5),
               "matched_dice": round(float(m.get("matched_dice", 0)), 5),
               "n_pred": int(m.get("n_pred", 0))}
        table.append(row)
        print(f"{f:<18} mAP50={row['mask_mAP50']:.4f} AP={row['mask_mAP50_95']:.4f} "
              f"dice={row['matched_dice']:.4f} n={row['n_pred']}", flush=True)

    rep = {"split": a.split, "images": len(files), "model": "SAM3",
           "n_classes": n_c, "table": table}
    json.dump(rep, open(os.path.join(a.out, "prompt_sweep.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    ok = [r for r in table if "error" not in r]
    if ok:
        base = next((r for r in ok if r["family"] == "species_only"), None)
        best = max(ok, key=lambda r: r["mask_mAP50"])
        print(f"\n最佳（mask_mAP50）: {best['family']} = {best['mask_mAP50']}")
        if base and base["mask_mAP50"]:
            print(f"相对纯物种名基线（{base['mask_mAP50']}）提升 "
                  f"{best['mask_mAP50'] / base['mask_mAP50']:.1f}×")
    print(f"DONE -> {a.out}/prompt_sweep.json")


if __name__ == "__main__":
    main()
