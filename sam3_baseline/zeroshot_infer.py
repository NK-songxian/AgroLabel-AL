"""SAM3 零样本预标注 → COCO 格式结果，供 eval_seg.py 评测。

走 Ultralytics 原生接口（`SAM3SemanticPredictor`），直接用 sam3.pt，
无需 facebookresearch/sam3 仓库与 torch 2.10。

协议（重要）：PlantSeg-34 每张图只含 1 个物种类别，因此本脚本默认走
`oracle-class` 提示：用该图真值类别名作文本提示，只测"分割质量上限"。
这不是可上线配置，而是与 YOLO 对比时的**上限基准**。

用法：
  python -m sam3_baseline.zeroshot_infer --split val \
      --data-root /root/autodl-tmp/Ryan/data_plantseg34 \
      --ckpt /root/autodl-tmp/Ryan/sam3.pt \
      --out /root/autodl-tmp/Ryan/runs/sam3_zeroshot/val --limit 200
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ckpt", required=True, help="本地 sam3.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0, help=">0 只跑前 N 张（冒烟）")
    ap.add_argument("--prompt-key", default="primary", choices=["primary", "name"])
    ap.add_argument("--bank", default="configs/species_prompt_bank.json")
    ap.add_argument("--fp32", action="store_true", help="默认 FP16")
    return ap.parse_args()


def load_bank(path):
    with open(path, encoding="utf-8") as f:
        return {c["class_id"]: c for c in json.load(f)["classes"]}


def read_label(path):
    ids, polys = [], []
    if not os.path.exists(path):
        return ids, polys
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 7:
                continue
            ids.append(int(float(p[0])))
            polys.append(np.array([float(x) for x in p[1:]], dtype=np.float32).reshape(-1, 2))
    return ids, polys


def polys_to_rle(polys, w, h):
    from pycocotools import mask as mask_utils

    m = np.zeros((h, w), dtype=np.uint8)
    d = ImageDraw.Draw(Image.fromarray(m))
    for poly in polys:
        xy = np.clip(poly * np.array([w, h], dtype=np.float32), 0, None)
        d.polygon([tuple(v) for v in xy.tolist()], fill=1)
    return mask_utils.encode(np.asfortranarray(m))


def rle_json(rle):
    """pycocotools RLE 的 counts 是 bytes，json 需转 ascii 字符串。"""
    c = rle["counts"]
    if isinstance(c, bytes):
        c = c.decode("ascii")
    return {"size": list(rle["size"]), "counts": c}


def build_predictor(ckpt, conf, half):
    from ultralytics.models.sam import SAM3SemanticPredictor

    ov = {
        "conf": conf,
        "task": "segment",
        "mode": "predict",
        "model": ckpt,
        "retina_masks": True,
        "verbose": False,
        "save": False,
    }
    if half:
        ov["quantize"] = 16
    return SAM3SemanticPredictor(overrides=ov)


def prompt_once(predictor, prompt):
    try:
        res = predictor(text=[prompt])
    except TypeError:
        res = predictor(text=prompt)
    if isinstance(res, list):
        return res[0] if res else None
    return res


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    from pycocotools import mask as mask_utils

    bank = load_bank(args.bank)
    predictor = build_predictor(args.ckpt, args.conf, not args.fp32)

    img_dir = os.path.join(args.data_root, "images", args.split)
    lbl_dir = os.path.join(args.data_root, "labels", args.split)
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT))
    if args.limit:
        files = files[: args.limit]

    gt = {"images": [], "annotations": [],
          "categories": [{"id": c + 1, "name": bank[c]["name"]} for c in sorted(bank)]}
    preds, ann_id, n_inst, n_zero, n_ok, t_all = [], 1, 0, 0, 0, 0.0

    for i, fn in enumerate(files):
        stem = os.path.splitext(fn)[0]
        with Image.open(os.path.join(img_dir, fn)) as im:
            im = im.convert("RGB")
            w, h = im.size
            img_id = i + 1
            gt["images"].append({"id": img_id, "file_name": fn, "width": w, "height": h})

            ids, polys = read_label(os.path.join(lbl_dir, stem + ".txt"))
            for cid, poly in zip(ids, polys):
                rle = polys_to_rle([poly], w, h)
                gt["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": cid + 1,
                    "segmentation": rle_json(rle), "area": float(mask_utils.area(rle)),
                    "bbox": mask_utils.toBbox(rle).tolist(), "iscrowd": 0})
                ann_id += 1

            if not ids:
                n_zero += 1
                continue
            cls_id = max(set(ids), key=ids.count)
            prompt = bank[cls_id][args.prompt_key]

            t0 = time.time()
            predictor.set_image(np.asarray(im))
            r = prompt_once(predictor, prompt)
            t_all += time.time() - t0

            if r is None or r.masks is None or len(r.masks) == 0:
                continue
            masks = r.masks.data.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy().reshape(-1)
            boxes = r.boxes.xyxy.cpu().numpy().reshape(-1, 4)
            n_inst += len(confs)
            n_ok += 1

            for m, b, s in zip(masks, boxes, confs):
                if float(s) < args.conf:
                    continue
                mm = m.astype(np.uint8)
                if mm.shape != (h, w):
                    mm = np.asarray(Image.fromarray(mm * 255).resize((w, h), Image.NEAREST)) > 127
                    mm = mm.astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mm))
                preds.append({
                    "image_id": img_id, "category_id": cls_id + 1,
                    "segmentation": rle_json(rle), "score": float(s),
                    "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})

        if (i + 1) % 50 == 0 or i + 1 == len(files):
            print(f"[{i+1}/{len(files)}] hit={n_ok} inst={n_inst} kept={len(preds)} "
                  f"avg={t_all/max(1,n_ok):.3f}s", flush=True)

    with open(os.path.join(args.out, "gt.json"), "w", encoding="utf-8") as f:
        json.dump(gt, f)
    with open(os.path.join(args.out, "pred.json"), "w", encoding="utf-8") as f:
        json.dump(preds, f)
    meta = {"split": args.split, "images": len(files), "no_label": n_zero,
            "images_hit": n_ok, "instances_raw": n_inst, "instances_kept": len(preds),
            "conf": args.conf, "prompt": args.prompt_key, "fp16": not args.fp32,
            "sec_total": round(t_all, 2), "sec_per_img": round(t_all / max(1, n_ok), 3),
            "ckpt": args.ckpt}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
