"""Task-SAM3-001 · 三、YOLO → COCO 转换（SAM3 训练/验证标注）

- 只用 train 生成训练标注，val 单独生成验证标注，**test 不转换**（保持独立）。
- 保留全部合法多边形；每个 annotation 附 noun_phrase。
- 记录 原图 ↔ 标签 ↔ COCO 的对应关系。
- 校验 polygon / bbox / area（多边形回栅格化比对）。
- 输出 20 张转换前后对比图。

用法：
  python datasets_tools/sam3_build_coco.py --data-root F:/_plantseg/dataset_plantseg_34 \
      --meta-root F:/_plantseg/dataset_plantseg_115
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw

IMG_EXT = (".jpg", ".jpeg", ".png")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="F:/_plantseg/dataset_plantseg_34")
    ap.add_argument("--meta-root", default="F:/_plantseg/dataset_plantseg_115")
    ap.add_argument("--out-dir", default="datasets/sam3")
    ap.add_argument("--preview-dir", default="experiments/sam3_conversion_preview")
    ap.add_argument("--preview-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--coord-decimals", type=int, default=2)
    return ap.parse_args()


def load_classes(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d if isinstance(d, list) else d["classes"]


def read_label(path):
    recs, bad = [], 0
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 7:
                bad += 1
                continue
            recs.append((int(float(p[0])), np.array([float(x) for x in p[1:]], np.float64).reshape(-1, 2)))
    return recs, bad


def shoelace(pl, w, h):
    x, y = pl[:, 0] * w, pl[:, 1] * h
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def raster_area(pl, w, h):
    canvas = Image.new("L", (w, h), 0)
    ImageDraw.Draw(canvas).polygon([tuple(v) for v in (pl * np.array([w, h])).tolist()], fill=1)
    return float(np.asarray(canvas).sum())


def verify_identity(coco, data_root, split, decimals=2):
    """全局严格校验：每条标注的绝对像素坐标，必须与原始 YOLO 标签逐一对应。"""
    by_img = defaultdict(list)
    for an in coco["annotations"]:
        by_img[an["image_id"]].append(tuple(np.round(np.asarray(an["segmentation"][0]), decimals)))
    ok, miss, extra = 0, 0, 0
    for im in coco["images"]:
        recs, _ = read_label(os.path.join(data_root, im["label_file"]))
        src = []
        for _, pl in recs:
            src.append(tuple(np.round((pl * np.array([im["width"], im["height"]])).reshape(-1), decimals)))
        got = by_img.get(im["id"], [])
        src_s, got_s = Counter(src), Counter(got)
        ok += sum((src_s & got_s).values())
        miss += sum((src_s - got_s).values())
        extra += sum((got_s - src_s).values())
    return {"matched": ok, "missing_in_coco": miss, "extra_in_coco": extra,
            "exact": miss == 0 and extra == 0}


def main():
    a = parse_args()
    random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.preview_dir, exist_ok=True)

    hosts = load_classes(os.path.join(a.data_root, "classes.json"))
    combos = load_classes(os.path.join(a.meta_root, "classes.json"))
    slug2 = sorted(((c.split("___")[1].lower().replace(" ", "_"), c.split("___")[0], c.split("___")[1])
                    for c in combos), key=lambda x: -len(x[0]))

    def match_disease(fn):
        s = os.path.splitext(fn)[0].lower()
        for slug, plant, dis in slug2:
            if s.startswith(slug):
                return plant, dis
        return None, None

    categories = [{"id": i + 1, "name": h, "host": h} for i, h in enumerate(hosts)]
    phrase_ids, phrases, cat_phrases = {}, [], {c["id"]: [] for c in categories}

    def pid(phrase, cat_id):
        if phrase not in phrase_ids:
            phrase_ids[phrase] = len(phrases) + 1
            phrases.append({"id": phrase_ids[phrase], "phrase": phrase, "category_id": cat_id})
            cat_phrases[cat_id].append(phrase)
        return phrase_ids[phrase]

    report, report_ident = {}, {}
    for split in ("train", "val"):
        img_dir = os.path.join(a.data_root, "images", split)
        lbl_dir = os.path.join(a.data_root, "labels", split)
        files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT))

        coco = {"info": {"task": "Task-SAM3-001", "split": split,
                         "source": f"YOLO-seg labels @ {a.data_root}/labels/{split}",
                         "note": "class=host plant, mask=disease lesion region; noun_phrase 由 115 类元数据的 Disease 字段生成",
                         "coord_decimals": a.coord_decimals,
                         "segmentation_unit": "absolute_pixel_xy"},
                "licenses": [], "categories": categories,
                "images": [], "annotations": [], "noun_phrases": []}
        ann_id, dropped, bad_lines = 1, 0, 0
        area_err, bbox_err = [], []

        for img_id, fn in enumerate(files, start=1):
            stem = os.path.splitext(fn)[0]
            with Image.open(os.path.join(img_dir, fn)) as im:
                W, H = im.size
            if W < 2 or H < 2:
                dropped += 1
                continue
            recs, bad = read_label(os.path.join(lbl_dir, stem + ".txt"))
            bad_lines += bad
            plant, dis = match_disease(fn)
            host = hosts[max(set(c for c, _ in recs), key=[c for c, _ in recs].count)] if recs else plant
            host = host or plant
            if host is None or host not in hosts:
                dropped += 1
                continue
            cat_id = hosts.index(host) + 1
            phrase = f"{dis.lower()} lesions on {host.lower()}" if dis else f"disease lesions on {host.lower()} leaves"
            np_id = pid(phrase, cat_id)

            coco["images"].append({
                "id": img_id, "file_name": fn, "width": W, "height": H, "split": split,
                "host": host, "disease": dis,
                "label_file": f"labels/{split}/{stem}.txt",
                "image_path": f"images/{split}/{fn}"})

            for cid, pl in recs:
                if len(pl) < 3:
                    dropped += 1
                    continue
                if not np.all((pl >= 0) & (pl <= 1)):
                    dropped += 1
                    continue
                xy = pl * np.array([W, H])
                seg = [round(float(v), a.coord_decimals) for v in xy.reshape(-1)]
                x1, y1 = float(xy[:, 0].min()), float(xy[:, 1].min())
                x2, y2 = float(xy[:, 0].max()), float(xy[:, 1].max())
                area = float(shoelace(pl, W, H))
                if area <= 1.0:
                    dropped += 1
                    continue
                ra = raster_area(pl, W, H)
                area_err.append(abs(ra - area) / max(area, 1.0))
                bbox_err.append((x1 < 0 or y1 < 0 or x2 > W + 0.5 or y2 > H + 0.5))
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": hosts.index(hosts[cid]) + 1,
                    "segmentation": [seg],
                    "area": round(area, 2),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "iscrowd": 0, "noun_phrase": phrase, "noun_phrase_id": np_id})
                ann_id += 1

        coco["noun_phrases"] = phrases
        for c in categories:
            c["noun_phrases"] = cat_phrases[c["id"]]

        ident = verify_identity(coco, a.data_root, split, a.coord_decimals)
        report_ident[split] = ident
        path = os.path.join(a.out_dir, f"plantseg34_{split}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)

        report[split] = {
            "file": path.replace("\\", "/"),
            "images": len(coco["images"]), "annotations": len(coco["annotations"]),
            "categories": len(categories), "noun_phrases": len(phrases),
            "dropped": dropped, "malformed_lines": bad_lines,
            "size_mb": round(os.path.getsize(path) / 1e6, 2),
            "area_rel_err": {"mean": round(float(np.mean(area_err)), 4) if area_err else 0,
                             "p95": round(float(np.percentile(area_err, 95)), 4) if area_err else 0,
                             "max": round(float(np.max(area_err)), 4) if area_err else 0},
            "bbox_out_of_image": int(sum(bbox_err)),
            "images_per_class": dict(Counter(i["host"] for i in coco["images"]).most_common()),
            "identity_check": ident,
        }
        print(f"[coco] {split}: {len(coco['images'])} 图 / {len(coco['annotations'])} 实例 / "
              f"{len(phrases)} 短语 / {report[split]['size_mb']} MB / 丢弃 {dropped}")

    # 20 张转换前后对比
    with open(os.path.join(a.out_dir, "plantseg34_train.json"), encoding="utf-8") as f:
        tr = json.load(f)
    img_by_id = {i["id"]: i for i in tr["images"]}
    anns_by_img = defaultdict(list)
    for an in tr["annotations"]:
        anns_by_img[an["image_id"]].append(an)
    picks = random.sample(tr["annotations"], min(a.preview_n, len(tr["annotations"])))
    rows = []
    for k, ann in enumerate(picks):
        im_info = img_by_id[ann["image_id"]]
        ipath = os.path.join(a.data_root, im_info["image_path"])
        im = Image.open(ipath).convert("RGB")
        W, H = im.size
        # 左：原始 YOLO 标签；右：从 COCO JSON 反解
        left = im.copy()
        dl = ImageDraw.Draw(left, "RGBA")
        recs, _ = read_label(os.path.join(a.data_root, im_info["label_file"]))
        for _, pl in recs:
            xy = [tuple(v) for v in (pl * np.array([W, H])).tolist()]
            dl.polygon(xy, fill=(255, 60, 60, 90), outline=(255, 0, 0, 255))
        right = im.copy()
        dr = ImageDraw.Draw(right, "RGBA")
        # 右侧画该图**全部** COCO 标注，保证与左侧同图可比；当前标注高亮
        for other in anns_by_img[ann["image_id"]]:
            poly = np.array(other["segmentation"][0], dtype=np.float64).reshape(-1, 2)
            cur = other["id"] == ann["id"]
            dr.polygon([tuple(v) for v in poly.tolist()],
                       fill=(60, 120, 255, 90) if not cur else (60, 255, 120, 110),
                       outline=(0, 90, 255, 255) if not cur else (0, 255, 120, 255))
        seg = np.array(ann["segmentation"][0]).reshape(-1, 2)
        x, y, w, h = ann["bbox"]
        dr.rectangle([x, y, x + w, y + h], outline=(0, 255, 120, 255), width=2)
        dr.text((6, 6), f"area={ann['area']:.0f} box={w:.0f}x{h:.0f}", fill=(255, 255, 0, 255))
        # 一致性：原始标签多边形 vs COCO 反解多边形，两张栅格图的 IoU
        def _mask(poly_abs):
            c = Image.new("L", (W, H), 0)
            ImageDraw.Draw(c).polygon([tuple(v) for v in poly_abs.tolist()], fill=1)
            return np.asarray(c, dtype=np.uint8) > 0

        m1 = _mask(seg.reshape(-1, 2))
        # 按坐标精确配对（同顶点数且逐点坐标一致才算同一条）
        flat = np.asarray(seg, dtype=np.float64).reshape(-1)

        def _key(pl):
            a = np.round(pl * np.array([W, H]), 2).reshape(-1)
            return 1e9 if a.shape != flat.shape else float(np.abs(a - flat).max())

        orig = min((pl for _, pl in recs), key=_key)
        m2 = _mask(orig * np.array([W, H]))
        inter, union = float((m1 & m2).sum()), float((m1 | m2).sum())
        iou = inter / union if union else 0.0
        rel = 1.0 - iou
        canvas = Image.new("RGB", (left.width + right.width + 8, H), (20, 20, 20))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width + 8, 0))
        out = os.path.join(a.preview_dir, f"{k:02d}_{im_info['file_name'].rsplit('.', 1)[0]}.jpg")
        canvas.save(out, quality=88)
        rows.append({"preview": out.replace("\\", "/"), "image": im_info["file_name"],
                     "host": im_info["host"], "disease": im_info["disease"],
                     "noun_phrase": ann["noun_phrase"], "area": ann["area"],
                     "bbox": ann["bbox"], "reparse_iou": round(iou, 5),
                     "reparse_rel_err": round(rel, 5)})

    max_rel = max(r["reparse_rel_err"] for r in rows)
    min_iou = min(r["reparse_iou"] for r in rows)
    summary = {"task": "Task-SAM3-001 · COCO 转换", "splits": report,
               "test_policy": "test 未转换，保持独立，不参与提示词或模型训练",
               "preview": {"n": len(rows), "dir": a.preview_dir.replace("\\", "/"),
                           "min_reparse_iou": round(min_iou, 5),
                           "mean_reparse_iou": round(float(np.mean([r["reparse_iou"] for r in rows])), 5),
                           "rows": rows}}
    with open(os.path.join(a.out_dir, "conversion_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[coco] 对比图 {len(rows)} 张 -> {a.preview_dir}，多边形回解最小 IoU {min_iou:.5f}")


if __name__ == "__main__":
    main()
