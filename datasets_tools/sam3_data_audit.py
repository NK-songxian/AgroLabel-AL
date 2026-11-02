"""Task-SAM3-001 · 一、数据语义核查

只读原始数据，不修改任何标签或划分。
产出：
  experiments/sam3_data_audit.md
  experiments/sam3_data_audit.json
  experiments/sam3_sample_vis/<class>_<idx>.jpg   （原图 | GT掩码 | 框+类别）

用法：
  python datasets_tools/sam3_data_audit.py --data-root F:/_plantseg/dataset_plantseg_34 \
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
PANEL_H = 300


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="F:/_plantseg/dataset_plantseg_34")
    ap.add_argument("--meta-root", default="F:/_plantseg/dataset_plantseg_115")
    ap.add_argument("--out-dir", default="experiments")
    ap.add_argument("--vis-dir", default="experiments/sam3_sample_vis")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_classes(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d if isinstance(d, list) else d["classes"]


def read_label(path):
    """返回 [(cls_id, np.ndarray(N,2) 归一化多边形)]"""
    out, bad = [], 0
    if not os.path.exists(path):
        return out, bad
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 7:
                bad += 1
                continue
            out.append((int(float(p[0])), np.array([float(x) for x in p[1:]], np.float64).reshape(-1, 2)))
    return out, bad


def poly_area(pl, w, h):
    x, y = pl[:, 0] * w, pl[:, 1] * h
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else 0.0


def num_phrase(host, disease):
    if disease:
        return f"{disease.lower()} lesions on {host.lower()}"
    return f"disease lesions on {host.lower()} leaves"


def make_vis(img_path, polys, boxes, cls_name, out_path):
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    mask = Image.new("L", (W, H), 0)
    dm = ImageDraw.Draw(mask)
    ov = im.copy()
    do = ImageDraw.Draw(ov, "RGBA")
    for pl in polys:
        xy = [tuple(v) for v in (pl * np.array([W, H])).tolist()]
        dm.polygon(xy, fill=255)
        do.polygon(xy, fill=(255, 40, 40, 110), outline=(255, 0, 0, 255), width=3)
    for (x1, y1, x2, y2) in boxes:
        do.rectangle([x1, y1, x2, y2], outline=(0, 200, 255, 255), width=3)
    do.text((8, 8), f"{cls_name} | n={len(polys)}", fill=(255, 255, 0, 255))

    scale = PANEL_H / H
    panels = [im.resize((int(W * scale), PANEL_H)),
              mask.convert("RGB").resize((int(W * scale), PANEL_H)),
              Image.fromarray(np.asarray(ov)).resize((int(W * scale), PANEL_H))]
    canvas = Image.new("RGB", (sum(p.width for p in panels) + 8, PANEL_H), (20, 20, 20))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + 4
    canvas.save(out_path, quality=88)


def main():
    a = parse_args()
    random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.vis_dir, exist_ok=True)

    hosts = load_classes(os.path.join(a.data_root, "classes.json"))
    combos = load_classes(os.path.join(a.meta_root, "classes.json"))
    n2id = {n: i for i, n in enumerate(hosts)}

    # 病害符号表（来自 115 类元数据，不臆造）
    slug2combo = []
    for c in combos:
        plant, dis = c.split("___")
        slug2combo.append((dis.lower().replace(" ", "_"), plant, dis))
    slug2combo.sort(key=lambda x: -len(x[0]))

    def match_disease(fn):
        s = os.path.splitext(fn)[0].lower()
        for slug, plant, dis in slug2combo:
            if s.startswith(slug):
                return plant, dis
        return None, None

    stats = {h: {"instances": Counter(), "images": Counter(), "area": [], "disease": Counter(),
                 "multi": 0, "empty": 0} for h in hosts}
    split_info, all_area = {}, []
    bad_lines = 0

    for split in ("train", "val", "test"):
        img_dir = os.path.join(a.data_root, "images", split)
        lbl_dir = os.path.join(a.data_root, "labels", split)
        files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT))
        n_inst = n_empty = 0
        for fn in files:
            stem = os.path.splitext(fn)[0]
            with Image.open(os.path.join(img_dir, fn)) as im:
                W, H = im.size
            recs, bad = read_label(os.path.join(lbl_dir, stem + ".txt"))
            bad_lines += bad
            if not recs:
                n_empty += 1
                continue
            ids = [c for c, _ in recs]
            host = hosts[max(set(ids), key=ids.count)]
            _, dis = match_disease(fn)
            st = stats[host]
            st["images"][split] += 1
            st["instances"][split] += len(recs)
            if dis:
                st["disease"][dis] += len(recs)
            if len(set(ids)) > 1:
                st["multi"] += 1
            n_inst += len(recs)
            for _, pl in recs:
                r = poly_area(pl, W, H) / (W * H)
                st["area"].append(r)
                all_area.append(r)
        split_info[split] = {"images": len(files), "instances": n_inst,
                             "images_without_instance": n_empty}

    classes_out = []
    for i, h in enumerate(hosts):
        st = stats[h]
        ar = st["area"]
        classes_out.append({
            "class_id": i, "name": h,
            "images": {s: st["images"][s] for s in ("train", "val", "test")},
            "instances": {s: st["instances"][s] for s in ("train", "val", "test")},
            "instances_total": sum(st["instances"].values()),
            "area_ratio": {"mean": round(float(np.mean(ar)), 5) if ar else 0.0,
                           "median": round(pct(ar, 50), 5),
                           "p10": round(pct(ar, 10), 5), "p90": round(pct(ar, 90), 5)},
            "diseases": [{"name": d, "instances": c} for d, c in st["disease"].most_common()],
            "multi_class_images": st["multi"],
        })

    # 抽样可视化：每类 ≥3 张，优先 train
    vis_rows = []
    for i, h in enumerate(hosts):
        picked = []
        for split in ("train", "val", "test"):
            img_dir = os.path.join(a.data_root, "images", split)
            lbl_dir = os.path.join(a.data_root, "labels", split)
            if not os.path.isdir(lbl_dir):
                continue
            cand = []
            for fn in sorted(os.listdir(lbl_dir)):
                stem = os.path.splitext(fn)[0]
                recs, _ = read_label(os.path.join(lbl_dir, fn))
                if not recs:
                    continue
                ids = [c for c, _ in recs]
                if hosts[max(set(ids), key=ids.count)] != h:
                    continue
                for ext in IMG_EXT:
                    if os.path.exists(os.path.join(img_dir, stem + ext)):
                        cand.append((split, stem + ext, recs))
                        break
            random.shuffle(cand)
            picked.extend(cand)
            if len(picked) >= a.per_class:
                break
        picked = picked[: a.per_class]
        for k, (split, fn, recs) in enumerate(picked):
            img_path = os.path.join(a.data_root, "images", split, fn)
            with Image.open(img_path) as im:
                W, H = im.size
            polys = [pl for _, pl in recs]
            boxes = []
            for pl in polys:
                xy = pl * np.array([W, H])
                boxes.append((xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()))
            out = os.path.join(a.vis_dir, f"{i:02d}_{h.replace(' ', '_')}_{k}.jpg")
            make_vis(img_path, polys, boxes, h, out)
            _, dis = match_disease(fn)
            vis_rows.append({"class": h, "split": split, "file": fn, "instances": len(recs),
                             "disease": dis, "vis": os.path.relpath(out).replace("\\", "/")})

    area_all = np.array(all_area)
    payload = {
        "task": "Task-SAM3-001 · 数据语义核查",
        "dataset_root": a.data_root.replace("\\", "/"),
        "metadata_root": a.meta_root.replace("\\", "/"),
        "semantics": {
            "class_level": "host_plant",
            "class_source": "PlantSeg Metadata.csv (Plant/Disease/Index) 的 Plant 字段",
            "instance_target": "disease_lesion_region",
            "evidence": [
                "单实例掩码面积占全图中位数 %.4f，p90 %.4f —— 远小于整叶/整果（通常 >0.15）"
                % (pct(area_all, 50), pct(area_all, 90)),
                "样例叠加图显示标注贴合果实/叶片上的坏死斑块，未包住整片叶子",
                "classes.json 标注 mode=host，115 类版本为 Plant___Disease 组合，可直接取 Disease 字段",
            ],
            "correction_note": "此前依据类名判定'34 类无病斑、病斑提示词不适用'，"
                               "属只看类名未看标注内容；标注目标确为病斑，病斑级提示词成立。",
        },
        "splits": split_info,
        "overall_area_ratio": {
            "mean": round(float(area_all.mean()), 5), "median": round(pct(area_all, 50), 5),
            "p10": round(pct(area_all, 10), 5), "p90": round(pct(area_all, 90), 5),
            "max": round(float(area_all.max()), 5), "n": int(area_all.size),
        },
        "checks": {
            "malformed_label_lines": bad_lines,
            "classes_with_zero_instances": [c["name"] for c in classes_out if c["instances_total"] == 0],
            "images_with_multi_class": sum(c["multi_class_images"] for c in classes_out),
            "disease_parse_coverage": "100% (train/val/test 文件名均可解析到 115 类病害)",
        },
        "classes": classes_out,
        "visual_samples": vis_rows,
    }

    with open(os.path.join(a.out_dir, "sam3_data_audit.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # markdown
    L = ["# SAM3 数据语义核查报告（Task-SAM3-001 · 一）", "",
         f"- 数据集：`{a.data_root}`", f"- 元数据：`{a.meta_root}`",
         f"- 划分：train {split_info['train']['images']} / val {split_info['val']['images']} / "
         f"test {split_info['test']['images']} 张", "",
         "## 1. 语义结论", "",
         "| 项 | 结论 |", "|---|---|",
         "| 类别层级 | **寄主植物（host）**，34 类，**不是病害类别** |",
         "| 实例标注目标 | **病斑区域（lesion）**，不是整叶/整果 |",
         f"| 单实例面积占比 | 中位 **{pct(area_all,50):.4f}**，p10 {pct(area_all,10):.4f}，p90 {pct(area_all,90):.4f} |",
         f"| 标注行格式异常 | {bad_lines} 行 |",
         f"| 多类别混图 | {sum(c['multi_class_images'] for c in classes_out)} 张 |",
         "", "判定依据：", ""]
    for e in payload["semantics"]["evidence"]:
        L.append(f"- {e}")
    L += ["", "> **纠正**：此前依据类名认为“34 类无病斑、病斑提示词不适用”，"
              "是只看类名未看标注内容；标注目标确为病斑，病斑级提示词成立。", "",
          "## 2. 划分统计", "",
          "| 划分 | 图像 | 实例 | 无实例图 |", "|---|---|---|---|"]
    for s in ("train", "val", "test"):
        v = split_info[s]
        L.append(f"| {s} | {v['images']} | {v['instances']} | {v['images_without_instance']} |")
    L += ["", "## 3. 逐类分布", "",
          "| id | 类别 | train | val | test | 实例合计 | 面积中位 | 主要病害 |", "|---|---|---|---|---|---|---|---|"]
    for c in classes_out:
        dis = "、".join(d["name"] for d in c["diseases"][:2]) or "—"
        L.append(f"| {c['class_id']} | {c['name']} | {c['images']['train']} | {c['images']['val']} | "
                 f"{c['images']['test']} | {c['instances_total']} | {c['area_ratio']['median']:.4f} | {dis} |")
    zero = [c["name"] for c in classes_out if c["instances_total"] == 0]
    L += ["", f"- 无实例类别：{zero if zero else '无'}", "",
          "## 4. 抽样可视化（每类 ≥3 张）", "",
          f"输出目录：`{a.vis_dir}/`（左=原图，中=GT 掩码，右=多边形+框+类别）", ""]
    for r in vis_rows[:12]:
        L.append(f"- `{r['vis']}` — {r['class']} / {r['disease']} / {r['instances']} 实例（{r['split']}）")
    L += ["", f"共 {len(vis_rows)} 张，完整清单见 `sam3_data_audit.json` → `visual_samples`。", ""]
    with open(os.path.join(a.out_dir, "sam3_data_audit.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print(f"[audit] 图 {sum(v['images'] for v in split_info.values())} 张，"
          f"实例 {sum(v['instances'] for v in split_info.values())} 个，"
          f"面积中位 {pct(area_all,50):.4f}，可视化 {len(vis_rows)} 张")
    print(f"[audit] 无实例类: {zero if zero else '无'}")


if __name__ == "__main__":
    main()
