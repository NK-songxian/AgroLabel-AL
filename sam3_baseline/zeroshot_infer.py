"""SAM3 零样本预标注 → COCO 格式结果，供 eval_seg.py 评测。

协议（重要）：PlantSeg-34 每张图只含 1 个物种类别，因此本脚本默认走
`oracle-class` 提示：用该图真值类别名作文本提示，只测"分割质量上限"。
这不是可直接上线的配置，而是与 YOLO 对比时的**上限基准**。

用法：
  python -m sam3_baseline.zeroshot_infer --split val \
      --data-root /root/autodl-tmp/Ryan/data_plantseg34 \
      --ckpt /root/autodl-tmp/Ryan/sam3.pt \
      --out runs/sam3_zeroshot/val
"""
import argparse
import json
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from PIL import Image, ImageDraw

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--data-root", required=True, help="含 images/{split} 与 labels/{split} 的目录")
    ap.add_argument("--ckpt", default="", help="本地 sam3.pt；留空则走 HF 自动下载")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 张，用于冒烟")
    ap.add_argument("--prompt-key", default="primary", choices=["primary", "name"],
                    help="primary=小写物种名；name=原始类名")
    ap.add_argument("--bank", default="configs/species_prompt_bank.json")
    return ap.parse_args()


def build_model(ckpt):
    """兼容不同版本的 build_sam3_image_model 签名。"""
    from sam3.model_builder import build_sam3_image_model
    import inspect

    sig = inspect.signature(build_sam3_image_model)
    kw = {}
    if ckpt:
        for k in ("ckpt_path", "checkpoint_path", "model_path", "ckpt"):
            if k in sig.parameters:
                kw[k] = ckpt
                break
        for k in ("load_from_HF",):
            if k in sig.parameters:
                kw[k] = False
    return build_sam3_image_model(**kw)


def load_bank(path):
    with open(path, encoding="utf-8") as f:
        bank = json.load(f)
    return {c["class_id"]: c for c in bank["classes"]}


def read_label(path):
    """返回 (class_ids, polygons_pixel)。多边形为归一化坐标，需乘宽高。"""
    ids, polys = [], []
    if not os.path.exists(path):
        return ids, polys
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 7:
                continue
            cid = int(float(p[0]))
            coords = np.array([float(x) for x in p[1:]], dtype=np.float32)
            ids.append(cid)
            polys.append(coords.reshape(-1, 2))
    return ids, polys


def polys_to_rle(polys, w, h):
    from pycocotools import mask as mask_utils

    m = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        xy = np.clip(poly * np.array([w, h], dtype=np.float32), 0, None)
        ImageDraw.Draw(Image.fromarray(m)).polygon([tuple(v) for v in xy.tolist()], fill=1)
    return mask_utils.encode(np.asfortranarray(m))


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    from sam3.model.sam3_image_processor import Sam3Processor
    from pycocotools import mask as mask_utils

    bank = load_bank(args.bank)
    model = build_model(args.ckpt)
    processor = Sam3Processor(model)

    img_dir = os.path.join(args.data_root, "images", args.split)
    lbl_dir = os.path.join(args.data_root, "labels", args.split)
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT))
    if args.limit:
        files = files[: args.limit]

    gt = {"images": [], "annotations": [], "categories": [
        {"id": cid + 1, "name": bank[cid]["name"]} for cid in sorted(bank)]}
    preds = []
    ann_id, n_inst, n_zero, t_all = 1, 0, 0, 0.0

    for i, fn in enumerate(files):
        stem = os.path.splitext(fn)[0]
        path = os.path.join(img_dir, fn)
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            img_id = i + 1
            gt["images"].append({"id": img_id, "file_name": fn, "width": w, "height": h})

            ids, polys = read_label(os.path.join(lbl_dir, stem + ".txt"))
            for cid, poly in zip(ids, polys):
                rle = polys_to_rle([poly], w, h)
                gt["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": cid + 1,
                    "segmentation": rle, "area": float(mask_utils.area(rle)),
                    "bbox": mask_utils.toBbox(rle).tolist(), "iscrowd": 0})
                ann_id += 1

            if not ids:
                n_zero += 1
                continue
            cls_id = max(set(ids), key=ids.count)
            prompt = bank[cls_id][args.prompt_key]

            t0 = time.time()
            state = processor.set_image(im)
            out = processor.set_text_prompt(state=state, prompt=prompt)
            t_all += time.time() - t0

            masks = out["masks"]
            boxes = out["boxes"]
            scores = out["scores"]
            if hasattr(masks, "cpu"):
                masks, boxes, scores = masks.cpu().numpy(), boxes.cpu().numpy(), scores.cpu().numpy()
            masks = np.asarray(masks)
            scores = np.asarray(scores).reshape(-1)
            boxes = np.asarray(boxes).reshape(-1, 4)
            n_inst += len(scores)

            for m, b, s in zip(masks, boxes, scores):
                if float(s) < args.conf:
                    continue
                rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
                preds.append({
                    "image_id": img_id, "category_id": cls_id + 1,
                    "segmentation": rle, "score": float(s),
                    "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})

        if (i + 1) % 100 == 0 or i + 1 == len(files):
            print(f"[{i+1}/{len(files)}] inst={n_inst} kept={len(preds)} "
                  f"avg={t_all/max(1,i+1-n_zero):.3f}s/img", flush=True)

    with open(os.path.join(args.out, "gt.json"), "w", encoding="utf-8") as f:
        json.dump(gt, f)
    with open(os.path.join(args.out, "pred.json"), "w", encoding="utf-8") as f:
        json.dump(preds, f)
    meta = {"split": args.split, "images": len(files), "no_label": n_zero,
            "instances_raw": n_inst, "instances_kept": len(preds), "conf": args.conf,
            "prompt": args.prompt_key, "sec_total": round(t_all, 2),
            "sec_per_img": round(t_all / max(1, len(files) - n_zero), 3),
            "ckpt": args.ckpt or "hf"}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
