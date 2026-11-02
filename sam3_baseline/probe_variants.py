"""变体提示探测：对指定类别逐条试 variants，看"换说法能不能救回来"。

  python -m sam3_baseline.probe_variants --data-root ... --ckpt ... \
      --only "Soybean,Rice,Tobacco,Zucchini,Potato,Grape,Plum,Peach,Ginger" --per-class 3
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
    ap.add_argument("--only", required=True, help="逗号分隔类名；all 表示全部")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=1008)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from ultralytics.models.sam import SAM3SemanticPredictor

    bank = {c["class_id"]: c for c in json.load(open(a.bank, encoding="utf-8"))["classes"]}
    name2cid = {c["name"]: cid for cid, c in bank.items()}
    targets = list(bank) if a.only == "all" else [name2cid[n.strip()] for n in a.only.split(",")]

    pred = SAM3SemanticPredictor(overrides=dict(
        conf=a.conf, task="segment", mode="predict", model=a.ckpt,
        imgsz=a.imgsz, retina_masks=True, quantize=16, verbose=False, save=False))

    img_dir = os.path.join(a.data_root, "images", a.split)
    lbl_dir = os.path.join(a.data_root, "labels", a.split)
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
    out = []
    for cid in targets:
        files = per_cls[cid]
        if not files:
            continue
        pick = random.sample(files, min(a.per_class, len(files)))
        for v in bank[cid]["variants"]:
            hit, mx = 0, 0.0
            for fn in pick:
                with Image.open(os.path.join(img_dir, fn)) as im:
                    arr = np.asarray(im.convert("RGB"))
                pred.set_image(arr)
                try:
                    r = pred(text=[v])[0]
                except Exception:  # noqa: BLE001
                    continue
                if r.masks is None or len(r.masks) == 0:
                    continue
                hit += 1
                mx = max(mx, float(r.boxes.conf.max()))
            out.append({"name": bank[cid]["name"], "prompt": v, "hit": hit,
                        "n_img": len(pick), "max_conf": round(mx, 3)})
            print(f"{bank[cid]['name']:<13} '{v}'  hit={hit}/{len(pick)}  max={mx:.3f}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    best = {}
    for r in out:
        if r["max_conf"] > best.get(r["name"], (0, ""))[0]:
            best[r["name"]] = (r["max_conf"], r["prompt"])
    print("\n== 每类最佳变体 ==")
    for k, (s, p) in sorted(best.items(), key=lambda x: -x[1][0]):
        print(f"  {k:<13} {p:<20} conf={s:.3f}")


if __name__ == "__main__":
    main()
