"""SAM3 逐类探测：每种物种取若干张，记录是否命中、实例数、最高分。

用于回答"提示词对哪些类有效、哪些类完全召不回"。
  python -m sam3_baseline.diag_probe --data-root ... --ckpt ... --per-class 4 --imgsz 1008
"""
import argparse
import json
import os
import random

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--bank", default="configs/species_prompt_bank.json")
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=1008)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from ultralytics.models.sam import SAM3SemanticPredictor

    bank = {c["class_id"]: c for c in json.load(open(a.bank, encoding="utf-8"))["classes"]}
    pred = SAM3SemanticPredictor(overrides=dict(
        conf=a.conf, task="segment", mode="predict", model=a.ckpt,
        imgsz=a.imgsz, retina_masks=True, quantize=16, verbose=False, save=False))

    img_dir = os.path.join(a.data_root, "images", a.split)
    lbl_dir = os.path.join(a.data_root, "labels", a.split)

    # 按类别建索引
    per_cls = {c: [] for c in bank}
    for fn in sorted(os.listdir(lbl_dir)):
        if not fn.endswith(".txt"):
            continue
        ids = []
        with open(os.path.join(lbl_dir, fn)) as f:
            for line in f:
                p = line.split()
                if p:
                    ids.append(int(float(p[0])))
        if not ids:
            continue
        stem = os.path.splitext(fn)[0]
        for ext in (".jpg", ".jpeg", ".png"):
            if os.path.exists(os.path.join(img_dir, stem + ext)):
                per_cls[max(set(ids), key=ids.count)].append(stem + ext)
                break

    random.seed(0)
    rows = []
    for cid in sorted(per_cls):
        files = per_cls[cid]
        if not files:
            rows.append({"class_id": cid, "name": bank[cid]["name"], "prompt": bank[cid]["primary"],
                         "n_img": 0, "hit": 0, "inst": 0, "max_conf": 0.0})
            continue
        pick = random.sample(files, min(a.per_class, len(files)))
        hit, inst, mx = 0, 0, 0.0
        for fn in pick:
            with Image.open(os.path.join(img_dir, fn)) as im:
                arr = np.asarray(im.convert("RGB"))
            pred.set_image(arr)
            try:
                r = pred(text=[bank[cid]["primary"]])[0]
            except Exception as e:  # noqa: BLE001
                print("ERR", fn, e, flush=True)
                continue
            if r.masks is None or len(r.masks) == 0:
                continue
            hit += 1
            n = len(r.boxes.conf)
            inst += n
            mx = max(mx, float(r.boxes.conf.max()))
        rows.append({"class_id": cid, "name": bank[cid]["name"], "prompt": bank[cid]["primary"],
                     "n_img": len(pick), "hit": hit, "inst": inst, "max_conf": round(mx, 3)})
        print(f"{bank[cid]['name']:<14} hit={hit}/{len(pick)} inst={inst} max={mx:.3f}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    tot = sum(r["hit"] for r in rows)
    n = sum(r["n_img"] for r in rows)
    print(f"\nTOTAL hit {tot}/{n} = {tot/max(1,n):.1%}  imgsz={a.imgsz}")


if __name__ == "__main__":
    main()
