# -*- coding: utf-8 -*-
"""
i04_decoupled_eval.py —— 病斑定位与宿主—病害识别解耦（矩阵 I-04）

三种口径，同一批图、同一 SAM3 权重，只换"类别信息怎么进模型"：

  host34      34 个宿主级提示（现有口径，与 Z1/F1 可比）
  direct115   115 个细粒度病名提示，一次性让模型自己挑（"直接 115 类"）
  decoupled   通用定位 + 上下文分类：先用一句泛化提示把"病斑"都框出来，
              再对每个掩码裁剪用 CLIP 在 34 个宿主名里做零样本分类
              —— 定位不问"是什么"，识别不问"在哪"，两条路各自更容易学

为什么要做这条对照
----------------
PlantSeg-34 的标签是"宿主级"（34 类），而原始数据其实带 115 个细粒度病名。
直接上 115 类时，类间差异常在**病征纹理**而非**宿主**上，文本提示容易打架；
反过来，只用 34 类又会丢掉"同宿主不同病"的区分力。decoupled 把这件事拆成两步，
是论文里"解耦"这条线的核心实验。

缺类处理（必须写清，否则数字对不上）
--------------------------------
val 缺 `Peach peach rust`、test 缺 `Coffee coffee black rot`：这两类在划分里没有图，
不能拿 0 分拉低平均值。脚本按"GT 里出现过才评估"处理，并在报告里列出
`classes_evaluated / classes_missing`。

用法
----
  python src/i04_decoupled_eval.py --mode decoupled --ckpt weights/sam3.pt \
      --data-root data/plantseg34 --meta-root datasets/plantseg115 \
      --split val --limit 100 --out runtime/logs/matrix/I-04/decoupled

  python src/i04_decoupled_eval.py --mode direct115 ... --max-classes 115
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import time

import numpy as np
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context
os.environ.setdefault("YOLO_OFFLINE", "1")

from pycocotools import mask as mask_utils  # noqa: E402

from sam3_baseline.eval_full import compute_metrics  # noqa: E402
from sam3_baseline.zeroshot_infer import (build_disease_lookup, load_json,  # noqa: E402
                                          match_disease, polys_to_rle, read_label,
                                          rle_json)


def parse_args():
    ap = argparse.ArgumentParser(description="定位—识别解耦评测（I-04）")
    ap.add_argument("--mode", required=True, choices=["host34", "direct115", "decoupled"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--meta-root", default="datasets/plantseg115")
    ap.add_argument("--species-bank", default="configs/species_prompt_bank_tuned.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=1008)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--max-masks", type=int, default=120)
    ap.add_argument("--max-classes", type=int, default=0, help="提示类数上限（0=全部）")
    ap.add_argument("--generic-prompts", default="plant disease lesion",
                    help="decoupled 模式的通用定位提示，逗号分隔")
    ap.add_argument("--clip", default="clip:ViT-B/32")
    ap.add_argument("--clip-batch", type=int, default=32)
    ap.add_argument("--nms-iou", type=float, default=0.6, help="decoupled 合并多提示后的掩码 NMS")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--boundary-max", type=int, default=300)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def mask_nms(items, iou_thr):
    """items: [(mask_uint8, score)]；按分数贪心去重。"""
    order = np.argsort([-s for _, s in items])
    keep = []
    for i in order:
        m, s = items[i]
        drop = False
        for j in keep:
            inter = float((m & items[j][0]).sum())
            if inter == 0:
                continue
            u = float(m.sum() + items[j][0].sum() - inter)
            if u > 0 and inter / u > iou_thr:
                drop = True
                break
        if not drop:
            keep.append(i)
    return [items[k] for k in keep]


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)

    import torch
    from ultralytics.models.sam import SAM3SemanticPredictor

    hosts = load_json(os.path.join(a.data_root, "classes.json"))
    hosts = hosts if isinstance(hosts, list) else hosts["classes"]

    sp = {c["class_id"]: c for c in load_json(a.species_bank)["classes"]}
    host_prompts = [sp[i].get("tuned_prompt") or sp[i]["primary"] for i in range(len(hosts))]

    cls115 = load_json(os.path.join(a.meta_root, "classes.json"))
    cls115 = cls115 if isinstance(cls115, list) else cls115["classes"]
    slug = build_disease_lookup(a.meta_root, a.data_root)

    img_dir = os.path.join(a.data_root, "images", a.split)
    lbl_dir = os.path.join(a.data_root, "labels", a.split)
    files = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if a.limit:
        files = files[: a.limit]

    prompt_list = hosts if a.mode == "host34" else cls115
    if a.max_classes:
        prompt_list = prompt_list[: a.max_classes]

    predictor = SAM3SemanticPredictor(overrides=dict(
        conf=a.conf, task="segment", mode="predict", model=a.ckpt,
        imgsz=a.imgsz, retina_masks=True, quantize=16, verbose=False, save=False))

    clip_model = clip_text = None
    if a.mode == "decoupled":
        from ultralytics.nn.text_model import build_text_model

        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        clip_model = build_text_model(a.clip, device=dev)
        clip_text = clip_model.encode_text(clip_model.tokenize(
            [f"a photo of {p}" for p in host_prompts]))
        print(f"[clip] {a.clip} 已加载，类别提示 {clip_text.shape}", flush=True)

    cats = hosts if a.mode == "host34" else (cls115 if a.mode == "direct115" else hosts)
    gt = {"images": [], "annotations": [],
          "categories": [{"id": i + 1, "name": c} for i, c in enumerate(cats)]}
    preds = []
    ann_id = 1
    skipped_unmapped = 0
    n_pred = 0
    t_all = 0.0

    for i, fn in enumerate(files):
        stem = os.path.splitext(fn)[0]
        with Image.open(os.path.join(img_dir, fn)) as im:
            im = im.convert("RGB")
            W, H = im.size
            img_id = i + 1

            ids, polys = read_label(os.path.join(lbl_dir, stem + ".txt"))

            # direct115：图名 → 115 类里的 id；对不上就整张跳过（缺类不评估）
            cat115 = None
            if a.mode == "direct115":
                plant, dis = match_disease(fn, slug)
                if plant is None or dis is None:
                    skipped_unmapped += 1
                    continue
                key = f"{plant}___{dis}"
                if key not in cls115:
                    skipped_unmapped += 1
                    continue
                cat115 = cls115.index(key) + 1

            gt["images"].append({"id": img_id, "file_name": fn, "width": W, "height": H})
            for j, poly in enumerate(polys):
                r = polys_to_rle([poly], W, H)
                # host34 / decoupled：逐实例用自己的宿主类；
                # direct115：一张图只有一个细粒度病名，所有实例共用
                cid = cat115 if cat115 is not None else (int(ids[j]) + 1)
                gt["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": int(cid),
                    "segmentation": rle_json(r), "area": float(mask_utils.area(r)),
                    "bbox": mask_utils.toBbox(r).tolist(), "iscrowd": 0})
                ann_id += 1

            t0 = time.time()
            predictor.set_image(np.asarray(im))
            arr = np.asarray(im)

            def run_prompt(name):
                try:
                    out = predictor(text=[name])[0]
                except Exception:  # noqa: BLE001
                    return []
                if out.masks is None or len(out.masks) == 0:
                    return []
                ms = out.masks.data.cpu().numpy()
                cf = out.boxes.conf.cpu().numpy().reshape(-1)
                bx = out.boxes.xyxy.cpu().numpy().reshape(-1, 4)
                if len(cf) > a.max_masks:
                    keep = np.argsort(-cf)[: a.max_masks]
                    ms, bx, cf = ms[keep], bx[keep], cf[keep]
                res = []
                for m, b, s in zip(ms, bx, cf):
                    if float(s) < a.conf:
                        continue
                    mm = (m > 0.5).astype(np.uint8)
                    if mm.shape != (H, W):
                        mm = (np.asarray(Image.fromarray(mm * 255).resize((W, H), Image.NEAREST)) > 127).astype(np.uint8)
                    if mm.sum() == 0:
                        continue
                    res.append((mm, float(s), [float(v) for v in b]))
                return res

            if a.mode in ("host34", "direct115"):
                for k, name in enumerate(prompt_list):
                    for mm, s, b in run_prompt(name):
                        rle = mask_utils.encode(np.asfortranarray(mm))
                        preds.append({"image_id": img_id, "category_id": k + 1,
                                      "segmentation": rle_json(rle), "score": s,
                                      "bbox": [float(v) for v in mask_utils.toBbox(rle)]})
                        n_pred += 1
            else:
                merged = []
                for gname in [x.strip() for x in a.generic_prompts.split(",") if x.strip()]:
                    merged += [(mm, s) for mm, s, _ in run_prompt(gname)]
                merged = mask_nms(merged, a.nms_iou)
                # 逐个掩码裁剪 → CLIP 分类到 34 宿主
                for mm, s in merged:
                    ys, xs = np.where(mm > 0)
                    if len(xs) < 16:
                        continue
                    x1, x2 = int(xs.min()), int(xs.max()) + 1
                    y1, y2 = int(ys.min()), int(ys.max()) + 1
                    crop = Image.fromarray(arr[y1:y2, x1:x2])
                    # 掩码外区域涂灰，避免背景主导 CLIP
                    sub = mm[y1:y2, x1:x2][..., None]
                    crop = Image.fromarray((arr[y1:y2, x1:x2] * sub + 127 * (1 - sub)).astype(np.uint8))
                    with torch.inference_mode():
                        feat = clip_model.encode_image(crop)
                        logits = (feat @ clip_text.t())[0]
                        cid = int(logits.argmax().item()) + 1
                    rle = mask_utils.encode(np.asfortranarray(mm))
                    preds.append({"image_id": img_id, "category_id": cid,
                                  "segmentation": rle_json(rle), "score": s,
                                  "bbox": [float(v) for v in mask_utils.toBbox(rle)]})
                    n_pred += 1

            t_all += time.time() - t0

        if (i + 1) % 20 == 0 or i + 1 == len(files):
            print(f"[{i+1}/{len(files)}] preds={n_pred} avg={t_all/max(1,i+1):.2f}s/img",
                  flush=True)

    json.dump(gt, open(os.path.join(a.out, "gt.json"), "w", encoding="utf-8"))
    json.dump(preds, open(os.path.join(a.out, "pred.json"), "w", encoding="utf-8"))

    present = {c["category_id"] for c in gt["annotations"]}
    evaluated = sorted(present)
    missing = [i + 1 for i in range(len(cats)) if (i + 1) not in present]

    m, _ = compute_metrics(gt, preds, a.boundary_max)
    rep = {
        "experiment": "I-04", "mode": a.mode, "split": a.split,
        "images": len(gt["images"]), "skipped_unmapped": skipped_unmapped,
        "n_classes": len(cats), "classes_evaluated": len(evaluated),
        "classes_missing": [cats[c - 1] for c in missing],
        "prompts_used": len(prompt_list),
        "avg_sec_per_image": round(t_all / max(1, len(gt["images"])), 3),
        "metrics": {k: m[k] for k in ("mask_mAP50", "mask_mAP50_95", "mask_mAP75",
                                      "instance_recall", "instance_coverage",
                                      "matched_dice", "boundary_f1",
                                      "n_pred", "n_matched")},
    }
    json.dump(rep, open(os.path.join(a.out, "i04_metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"\n=== I-04 {a.mode} ===")
    for k, v in rep["metrics"].items():
        print(f"{k:>18}: {v}")
    print(f"{'avg_sec/img':>18}: {rep['avg_sec_per_image']}")
    print(f"{'classes_missing':>18}: {rep['classes_missing']}")
    print(f"DONE -> {a.out}")


if __name__ == "__main__":
    main()
