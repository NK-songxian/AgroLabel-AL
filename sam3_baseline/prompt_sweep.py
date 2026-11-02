"""提示词扫描（S0/S1）：一次编码，多次文本提示，统一六项指标。

变体：
  species_raw     裸类名（S0）
  species_tuned   每类最优物种名（S1-species）
  host_0..host_3  host_prompt_bank 寄主级短语
  disease_0..2    病害级短语

同一张图只做一次 set_image，随后对所有变体复用该编码，成本与变体数基本无关。

用法：
  python -m sam3_baseline.prompt_sweep --split val --data-root ... --ckpt ... \
      --out runs/prompt_sweep
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from sam3_baseline.eval_full import compute_metrics
from sam3_baseline.zeroshot_infer import (build_disease_lookup, load_json,
                                          polys_to_rle, rle_json, match_disease,
                                          read_label)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--imgsz", type=int, default=1008)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--boundary-max", type=int, default=300)
    ap.add_argument("--host-bank", default="configs/sam3/host_prompt_bank.json")
    ap.add_argument("--species-bank", default="configs/species_prompt_bank_tuned.json")
    ap.add_argument("--variants", default="species_tuned,host_0,disease_0,disease_1")
    ap.add_argument("--max-masks", type=int, default=120, help="每图每变体最多保留的实例数")
    return ap.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    from ultralytics.models.sam import SAM3SemanticPredictor

    hosts = load_json(os.path.join(a.data_root, "classes.json"))
    hosts = hosts if isinstance(hosts, list) else hosts["classes"]
    sp = {c["class_id"]: c for c in load_json(a.species_bank)["classes"]}
    hb = {h["host"]: h for h in load_json(a.host_bank)["hosts"]}
    slug = build_disease_lookup(a.data_root.replace("34", "115"), a.data_root)

    predictor = SAM3SemanticPredictor(overrides=dict(
        conf=a.conf, task="segment", mode="predict", model=a.ckpt,
        imgsz=a.imgsz, retina_masks=True, quantize=16, verbose=False, save=False))

    img_dir = os.path.join(a.data_root, "images", a.split)
    lbl_dir = os.path.join(a.data_root, "labels", a.split)
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if a.limit:
        files = files[: a.limit]

    VARIANT = [v.strip() for v in a.variants.split(",") if v.strip()]
    preds = {v: [] for v in VARIANT}
    prompt_used = {v: {} for v in VARIANT}

    gt = {"images": [], "annotations": [],
          "categories": [{"id": i + 1, "name": h} for i, h in enumerate(hosts)]}
    ann_id, t_all, n_hit = 1, 0.0, 0
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
                gt["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": cid + 1,
                                          "segmentation": rle_json(r),
                                          "area": float(mask_utils.area(r)),
                                          "bbox": mask_utils.toBbox(r).tolist(), "iscrowd": 0})
                ann_id += 1
            if not ids:
                continue
            cid = max(set(ids), key=ids.count)
            host = hosts[cid]
            _, dis = match_disease(fn, slug)
            e = sp[cid]
            h = hb[host]

            prompts = {
                "species_raw": e["name"].lower(),
                "species_tuned": e.get("tuned_prompt") or e["primary"],
            }
            for k in range(4):
                prompts[f"host_{k}"] = h["host_level_phrases"][min(k, len(h["host_level_phrases"]) - 1)]
            dph = None
            for d in h["diseases"]:
                if dis and d["name"].lower() == dis.lower():
                    dph = d["phrases"]
                    break
            for k in range(3):
                prompts[f"disease_{k}"] = (dph[min(k, len(dph) - 1)] if dph
                                           else prompts["host_0"])

            t0 = time.time()
            predictor.set_image(np.asarray(im))
            t_all += time.time() - t0
            n_hit += 1
            for v in VARIANT:
                pr = prompts[v]
                prompt_used[v][pr] = prompt_used[v].get(pr, 0) + 1
                try:
                    out = predictor(text=[pr])[0]
                except Exception:  # noqa: BLE001
                    continue
                if out.masks is None or len(out.masks) == 0:
                    continue
                ms = out.masks.data.cpu().numpy()
                cf = out.boxes.conf.cpu().numpy().reshape(-1)
                bx = out.boxes.xyxy.cpu().numpy().reshape(-1, 4)
                if len(cf) > a.max_masks:
                    keep = np.argsort(-cf)[: a.max_masks]
                    ms, bx, cf = ms[keep], bx[keep], cf[keep]
                for m, b, s in zip(ms, bx, cf):
                    if float(s) < a.conf:
                        continue
                    mm = m.astype(np.uint8)
                    if mm.shape != (H, W):
                        mm = (np.asarray(Image.fromarray(mm * 255).resize((W, H), Image.NEAREST)) > 127).astype(np.uint8)
                    preds[v].append({"image_id": img_id, "category_id": cid + 1,
                                     "segmentation": rle_json(mask_utils.encode(np.asfortranarray(mm))),
                                     "score": float(s),
                                     "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            print(f"[{i+1}/{len(files)}] enc_avg={t_all/max(1,n_hit):.3f}s "
                  f"preds={{{', '.join(f'{v}:{len(preds[v])}' for v in VARIANT)}}}", flush=True)

    json.dump(gt, open(os.path.join(a.out, "gt.json"), "w", encoding="utf-8"))
    table = []
    for v in VARIANT:
        json.dump(preds[v], open(os.path.join(a.out, f"pred_{v}.json"), "w", encoding="utf-8"))
        m, _ = compute_metrics(gt, preds[v], a.boundary_max)
        row = {"variant": v, **{k: m[k] for k in ("mask_mAP50", "mask_mAP50_95", "instance_recall",
                                                 "instance_coverage", "matched_dice", "boundary_f1",
                                                 "n_pred", "n_matched")}}
        table.append(row)
        print(f"{v:<16} mAP50={m['mask_mAP50']:.4f} AP={m['mask_mAP50_95']:.4f} "
              f"R={m['instance_recall']:.4f} cov={m['instance_coverage']:.4f} "
              f"dice={m['matched_dice']:.4f} bF1={m['boundary_f1']:.4f} n={m['n_pred']}", flush=True)

    json.dump({"split": a.split, "images": len(files), "table": table,
               "prompt_used": {k: dict(sorted(v.items(), key=lambda x: -x[1])[:6])
                               for k, v in prompt_used.items()}},
              open(os.path.join(a.out, "sweep.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    best = max(table, key=lambda r: r["mask_mAP50"])
    print(f"\n最佳（mAP50）: {best['variant']}  {best['mask_mAP50']:.4f}")


if __name__ == "__main__":
    main()
