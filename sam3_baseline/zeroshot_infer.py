"""SAM3 提示词推理（S0/S1 基线）→ COCO 结果，供 eval_full.py 评测。

三种提示来源：
  species  旧口径：物种名 / tuned 物种名（S0、S1-species，目标错配，仅留基线）
  host     host_prompt_bank 的寄主级短语（disease lesions on {host} leaves 等）
  disease  host_prompt_bank 的病害级短语（{disease} lesions on {host} …）

每张图只用其自身 host（及自身 disease）的短语，不做 34 类遍历。

用法：
  python -m sam3_baseline.zeroshot_infer --split val --data-root ... --ckpt ... \
      --out runs/s1_host0 --prompt-source host --phrase-index 0
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=1008)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--prompt-source", default="species", choices=["species", "host", "disease"])
    ap.add_argument("--phrase-index", type=int, default=0)
    ap.add_argument("--prompt-key", default="primary", choices=["primary", "name", "tuned"])
    ap.add_argument("--bank", default="configs/species_prompt_bank_tuned.json")
    ap.add_argument("--host-bank", default="configs/sam3/host_prompt_bank.json")
    ap.add_argument("--meta-root", default="", help="115 类 classes.json 所在目录")
    return ap.parse_args()


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_species_bank(path):
    return {c["class_id"]: c for c in load_json(path)["classes"]}


def build_disease_lookup(meta_root, data_root):
    classes = load_json(os.path.join(meta_root, "classes.json"))
    classes = classes if isinstance(classes, list) else classes["classes"]
    slug = sorted(((c.split("___")[1].lower().replace(" ", "_"), c.split("___")[0], c.split("___")[1])
                   for c in classes), key=lambda x: -len(x[0]))
    return slug


def match_disease(fn, slug):
    s = os.path.splitext(fn)[0].lower()
    for sl, plant, dis in slug:
        if s.startswith(sl):
            return plant, dis
    return None, None


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
            polys.append(np.array([float(x) for x in p[1:]], np.float32).reshape(-1, 2))
    return ids, polys


def polys_to_rle(polys, w, h):
    from pycocotools import mask as mask_utils

    canvas = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(canvas)
    for poly in polys:
        xy = np.clip(poly * np.array([w, h], dtype=np.float32), 0, None)
        if len(xy) >= 3:
            d.polygon([tuple(v) for v in xy.tolist()], fill=1)
    return mask_utils.encode(np.asfortranarray(np.asarray(canvas, dtype=np.uint8)))


def rle_json(rle):
    c = rle["counts"]
    if isinstance(c, bytes):
        c = c.decode("ascii")
    return {"size": list(rle["size"]), "counts": c}


def build_predictor(ckpt, conf, half, imgsz):
    from ultralytics.models.sam import SAM3SemanticPredictor

    ov = {"conf": conf, "task": "segment", "mode": "predict", "model": ckpt,
          "imgsz": imgsz, "retina_masks": True, "verbose": False, "save": False}
    if half:
        ov["quantize"] = 16
    return SAM3SemanticPredictor(overrides=ov)


def prompt_once(predictor, prompt):
    try:
        res = predictor(text=[prompt])
    except TypeError:
        res = predictor(text=prompt)
    return res[0] if isinstance(res, list) and res else res


def resolve_prompt(args, cls_id, host, disease, sp_bank, host_bank_by_name):
    if args.prompt_source == "species":
        e = sp_bank[cls_id]
        return e.get(args.prompt_key) or e["primary"]
    hb = host_bank_by_name[host]
    if args.prompt_source == "host":
        ph = hb["host_level_phrases"]
        return ph[min(args.phrase_index, len(ph) - 1)]
    if not disease:
        return hb["host_level_phrases"][0]
    for d in hb["diseases"]:
        if d["name"].lower() == disease.lower():
            ph = d["phrases"]
            return ph[min(args.phrase_index, len(ph) - 1)]
    return hb["host_level_phrases"][0]


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    from pycocotools import mask as mask_utils

    sp_bank = load_species_bank(args.bank)
    host_bank = load_json(args.host_bank)
    hosts = load_json(os.path.join(args.data_root, "classes.json"))
    hosts = hosts if isinstance(hosts, list) else hosts["classes"]
    host_bank_by_name = {h["host"]: h for h in host_bank["hosts"]}
    slug = build_disease_lookup(args.meta_root, args.data_root)

    predictor = build_predictor(args.ckpt, args.conf, not args.fp32, args.imgsz)
    img_dir = os.path.join(args.data_root, "images", args.split)
    lbl_dir = os.path.join(args.data_root, "labels", args.split)
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT))
    if args.limit:
        files = files[: args.limit]

    gt = {"images": [], "annotations": [],
          "categories": [{"id": i + 1, "name": h} for i, h in enumerate(hosts)]}
    preds, ann_id, n_inst, n_hit, t_all, used = [], 1, 0, 0, 0.0, {}
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
                gt["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": cid + 1,
                                          "segmentation": rle_json(rle),
                                          "area": float(mask_utils.area(rle)),
                                          "bbox": mask_utils.toBbox(rle).tolist(), "iscrowd": 0})
                ann_id += 1
            if not ids:
                continue
            cls_id = max(set(ids), key=ids.count)
            host = hosts[cls_id]
            _, disease = match_disease(fn, slug)
            prompt = resolve_prompt(args, cls_id, host, disease, sp_bank, host_bank_by_name)
            used[prompt] = used.get(prompt, 0) + 1

            t0 = time.time()
            predictor.set_image(np.asarray(im))
            r = prompt_once(predictor, prompt)
            t_all += time.time() - t0
            if r is None or r.masks is None or len(r.masks) == 0:
                continue
            n_hit += 1
            masks = r.masks.data.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy().reshape(-1)
            boxes = r.boxes.xyxy.cpu().numpy().reshape(-1, 4)
            n_inst += len(confs)
            for m, b, s in zip(masks, boxes, confs):
                if float(s) < args.conf:
                    continue
                mm = m.astype(np.uint8)
                if mm.shape != (h, w):
                    mm = (np.asarray(Image.fromarray(mm * 255).resize((w, h), Image.NEAREST)) > 127).astype(np.uint8)
                preds.append({"image_id": img_id, "category_id": cls_id + 1,
                              "segmentation": rle_json(mask_utils.encode(np.asfortranarray(mm))),
                              "score": float(s),
                              "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
        if (i + 1) % 200 == 0 or i + 1 == len(files):
            print(f"[{i+1}/{len(files)}] hit={n_hit} inst={n_inst} kept={len(preds)} "
                  f"avg={t_all/max(1,n_hit):.3f}s", flush=True)

    with open(os.path.join(args.out, "gt.json"), "w", encoding="utf-8") as f:
        json.dump(gt, f)
    with open(os.path.join(args.out, "pred.json"), "w", encoding="utf-8") as f:
        json.dump(preds, f)
    meta = {"split": args.split, "images": len(files), "images_hit": n_hit,
            "instances_raw": n_inst, "instances_kept": len(preds), "conf": args.conf,
            "imgsz": args.imgsz, "prompt_source": args.prompt_source,
            "phrase_index": args.phrase_index, "fp16": not args.fp32,
            "sec_total": round(t_all, 2), "sec_per_img": round(t_all / max(1, n_hit), 3),
            "ckpt": args.ckpt, "distinct_prompts": len(used),
            "prompt_preview": dict(sorted(used.items(), key=lambda x: -x[1])[:5])}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
