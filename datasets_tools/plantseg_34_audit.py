#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task-002A: PlantSeg 34 类数据正式审计  (只读，不修改任何数据)
================================================================
检查 dataset_plantseg_34 的 data.yaml / classes.json，统计 train/val/test
逐类别图片数 & 实例数，确认 34 类全覆盖，计算 AL 初始标注集 al34_random
占 train 比例、与 val/test 重叠、未标注池 U0，并诊断类别不平衡。

输出:
  experiments/plantseg_34_audit.md
  experiments/plantseg_34_audit.json
  experiments/class_distribution.csv

严格约束: 仅读取，不改数据/标签/划分，不下载，所有统计来自真实文件。
运行: E:\anaco\envs\sam_gpu\python.exe datasets_tools/plantseg_34_audit.py
"""

import os
import csv
import json
from collections import defaultdict

DATA_ROOT = r"F:\_plantseg"
DATASET_DIR = r"F:\_plantseg\dataset_plantseg_34"
AL_DIR = r"F:\_plantseg\_al\al34_random"   # 注: 实际目录为 al34_random (非 a134_random)
OUT_DIR = r"F:\AgroLabel-AL\experiments"
SPLITS = ["train", "val", "test"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(d):
    if not os.path.isdir(d):
        return []
    return [f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)]


def list_labels(d):
    if not os.path.isdir(d):
        return []
    return [f for f in os.listdir(d) if f.lower().endswith(".txt")]


def parse_label_classes(p):
    """返回该标签文件中出现的 {class_id: instance_count}"""
    out = defaultdict(int)
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                toks = line.split()
                if not toks:
                    continue
                try:
                    cid = int(toks[0])
                except ValueError:
                    continue
                out[cid] += 1
    except Exception as e:
        print(f"  [WARN] 读取标签失败 {p}: {e}")
    return out


def main():
    report = {
        "task": "Task-002A PlantSeg 34类数据正式审计",
        "constraint": "只读审计，未修改任何数据/标签/划分",
        "paths": {
            "data_root": DATA_ROOT,
            "dataset_dir": DATASET_DIR,
            "data_yaml": os.path.join(DATASET_DIR, "data.yaml"),
            "classes_json": os.path.join(DATASET_DIR, "classes.json"),
            "al_dir": AL_DIR,
            "al_labeled_txt": os.path.join(AL_DIR, "labeled.txt"),
        },
    }

    # ---------- 1. 配置文件检查 ----------
    cfg = {}
    # classes.json
    names = []
    cj_path = report["paths"]["classes_json"]
    if os.path.exists(cj_path):
        with open(cj_path, encoding="utf-8") as f:
            cj = json.load(f)
        names = cj.get("classes", [])
        cfg["classes_json"] = {
            "exists": True,
            "n_classes": len(names),
            "mode": cj.get("mode"),
            "source": cj.get("source"),
            "note": cj.get("note"),
        }
    else:
        cfg["classes_json"] = {"exists": False}
    # data.yaml (nc + names)
    dy_path = report["paths"]["data_yaml"]
    yaml_nc = None
    yaml_names = []
    if os.path.exists(dy_path):
        with open(dy_path, encoding="utf-8") as f:
            for line in f:
                s = line.rstrip("\n")
                if s.startswith("nc:"):
                    try:
                        yaml_nc = int(s.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif ":" in s:
                    key = s.split(":", 1)[0].strip()
                    if key.isdigit():
                        yaml_names.append(s.split(":", 1)[1].strip())
        cfg["data_yaml"] = {
            "exists": True,
            "path_field": None,  # 简单扫描
            "nc": yaml_nc,
            "names_count": len(yaml_names),
            "names_match_classes_json": (yaml_names == names),
        }
        # 扫描 path / train / val / test 字段
        with open(dy_path, encoding="utf-8") as f:
            for line in f:
                s = line.rstrip("\n")
                if s.startswith("path:"):
                    cfg["data_yaml"]["path_field"] = s.split(":", 1)[1].strip()
    else:
        cfg["data_yaml"] = {"exists": False}
    report["config_check"] = cfg

    nc = len(names)
    report["n_classes"] = nc

    # ---------- 2/3/4. 逐类别 图片数 & 实例数 (train/val/test) ----------
    # per_class[split][cid] = {"imgs": int, "inst": int}
    per_class = {sp: defaultdict(lambda: {"imgs": 0, "inst": 0}) for sp in SPLITS}
    split_totals = {}
    missing_classes = {sp: [] for sp in SPLITS}
    empty_label_imgs = {sp: 0 for sp in SPLITS}

    for sp in SPLITS:
        img_dir = os.path.join(DATASET_DIR, "images", sp)
        lbl_dir = os.path.join(DATASET_DIR, "labels", sp)
        imgs = list_images(img_dir)
        lbls = {os.path.splitext(f)[0]: f for f in list_labels(lbl_dir)}
        split_totals[sp] = {"n_images": len(imgs), "n_labels": len(lbls)}

        # 空标签检测
        for base in imgs:
            b = os.path.splitext(base)[0]
            if b not in lbls:
                continue  # 缺失标签另行统计(此处仅计空标签)
        # 逐标签统计
        for base, lf in lbls.items():
            cls_counts = parse_label_classes(os.path.join(lbl_dir, lf))
            if not cls_counts:
                empty_label_imgs[sp] += 1
            for cid, n in cls_counts.items():
                per_class[sp][cid]["inst"] += n
                per_class[sp][cid]["imgs"] += 1  # 该图含此类的 1 个实例 => 计入该类图片数

        # 缺失类
        for cid in range(nc):
            if per_class[sp][cid]["inst"] == 0:
                missing_classes[sp].append(cid)

    report["split_totals"] = split_totals
    report["empty_label_images"] = empty_label_imgs
    report["missing_classes"] = {
        sp: [{"id": c, "name": names[c]} for c in missing_classes[sp]]
        for sp in SPLITS
    }
    report["all_classes_present"] = all(len(missing_classes[sp]) == 0 for sp in SPLITS)

    # ---------- 4. 逐类别分布表 ----------
    dist = []
    total_inst_all = {sp: 0 for sp in SPLITS}
    for cid in range(nc):
        row = {
            "class_id": cid,
            "class_name": names[cid],
            "train_imgs": per_class["train"][cid]["imgs"],
            "train_inst": per_class["train"][cid]["inst"],
            "val_imgs": per_class["val"][cid]["imgs"],
            "val_inst": per_class["val"][cid]["inst"],
            "test_imgs": per_class["test"][cid]["imgs"],
            "test_inst": per_class["test"][cid]["inst"],
        }
        row["total_imgs"] = row["train_imgs"] + row["val_imgs"] + row["test_imgs"]
        row["total_inst"] = row["train_inst"] + row["val_inst"] + row["test_inst"]
        total_inst_all["train"] += row["train_inst"]
        total_inst_all["val"] += row["val_inst"]
        total_inst_all["test"] += row["test_inst"]
        dist.append(row)
    report["class_distribution"] = dist
    report["total_instances"] = total_inst_all

    # ---------- 5. 图片总数 (已在 split_totals) ----------
    total_imgs_all = sum(split_totals[sp]["n_images"] for sp in SPLITS)
    report["total_images"] = total_imgs_all

    # ---------- 6. al34_random 占 train 比例 ----------
    al_imgs = list_images(os.path.join(AL_DIR, "images", "train"))
    al_lbls = list_labels(os.path.join(AL_DIR, "labels", "train"))
    al_set = {os.path.splitext(f)[0] for f in al_imgs}
    train_imgs_set = {os.path.splitext(f)[0] for f in list_images(os.path.join(DATASET_DIR, "images", "train"))}

    al_stat = {
        "n_images": len(al_imgs),
        "n_labels": len(al_lbls),
        "train_total": len(train_imgs_set),
        "proportion_of_train": round(len(al_set) / len(train_imgs_set), 4) if train_imgs_set else None,
        "is_subset_of_train": al_set.issubset(train_imgs_set),
        "not_in_train": sorted(al_set - train_imgs_set),
    }
    # labeled.txt 一致性
    lt = report["paths"]["al_labeled_txt"]
    labeled_set = set()
    if os.path.exists(lt):
        with open(lt, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                labeled_set.add(os.path.splitext(os.path.basename(line))[0])
    al_stat["labeled_txt_unique"] = len(labeled_set)
    al_stat["in_al_not_in_labeled"] = sorted(al_set - labeled_set)
    al_stat["in_labeled_not_in_al"] = sorted(labeled_set - al_set)
    report["al_subset"] = al_stat

    # ---------- 7. 与 val/test 重叠 ----------
    val_set = {os.path.splitext(f)[0] for f in list_images(os.path.join(DATASET_DIR, "images", "val"))}
    test_set = {os.path.splitext(f)[0] for f in list_images(os.path.join(DATASET_DIR, "images", "test"))}
    overlap_val = sorted(al_set & val_set)
    overlap_test = sorted(al_set & test_set)
    report["al_overlap"] = {
        "with_val": overlap_val,
        "with_test": overlap_test,
        "no_overlap_with_val_test": (len(overlap_val) == 0 and len(overlap_test) == 0),
    }

    # ---------- 8. 未标注池 U0 ----------
    u0 = sorted(train_imgs_set - al_set)
    report["unlabeled_pool_U0"] = {
        "size": len(u0),
        "formula": "U0 = |train| - |al34_random| = %d - %d = %d" % (
            len(train_imgs_set), len(al_set), len(u0)),
    }

    # ---------- 9. 类别不平衡诊断 ----------
    inst_list = [d["total_inst"] for d in dist]
    img_list = [d["total_imgs"] for d in dist]
    train_inst_list = [d["train_inst"] for d in dist]
    n = len(inst_list)
    mean_i = sum(inst_list) / n
    mean_t = sum(train_inst_list) / n

    def stats(lst):
        s = sorted(lst)
        mn, mx = s[0], s[-1]
        mid = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
        var = sum((x - mean_i) ** 2 for x in lst) / n if lst is inst_list else sum((x - mean_t) ** 2 for x in lst) / n
        import math
        std = math.sqrt(var)
        cv = (std / mean_i) if (lst is inst_list and mean_i) else (std / mean_t if mean_t else 0)
        return mn, mx, mid, std, cv

    imn, imx, imid, ist, icv = stats(inst_list)
    tmn, tmx, tmid, tst, tcv = stats(train_inst_list)
    # 阈值: 实例 < 30 视为稀有; 占比 < 0.5% 视为极低
    total_inst = sum(inst_list)
    rare_inst = [d["class_name"] for d in dist if d["total_inst"] < 30]
    low_share = [d["class_name"] for d in dist if total_inst and d["total_inst"] / total_inst < 0.005]
    imbalance = {
        "by_total_instance": {
            "min": imn, "max": imx, "median": imid, "mean": round(mean_i, 1),
            "std": round(ist, 1), "cv": round(icv, 3), "max_min_ratio": round(imx / imn, 2) if imn else None,
        },
        "by_train_instance": {
            "min": tmn, "max": tmx, "median": tmid, "mean": round(mean_t, 1),
            "std": round(tst, 1), "cv": round(tcv, 3), "max_min_ratio": round(tmx / tmn, 2) if tmn else None,
        },
        "rare_classes_total_inst_lt_30": rare_inst,
        "very_low_share_lt_0.5pct": low_share,
        "extreme_imbalance_flag": (imx / imn > 10) if imn else True,
    }
    report["class_imbalance"] = imbalance

    # ---------- 写出 JSON ----------
    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, "plantseg_34_audit.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON  -> {json_path}")

    # ---------- 写出 CSV ----------
    csv_path = os.path.join(OUT_DIR, "class_distribution.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name",
                    "train_imgs", "train_inst",
                    "val_imgs", "val_inst",
                    "test_imgs", "test_inst",
                    "total_imgs", "total_inst"])
        for d in dist:
            w.writerow([d["class_id"], d["class_name"],
                        d["train_imgs"], d["train_inst"],
                        d["val_imgs"], d["val_inst"],
                        d["test_imgs"], d["test_inst"],
                        d["total_imgs"], d["total_inst"]])
    print(f"[OK] CSV   -> {csv_path}")

    # ---------- 写出 Markdown ----------
    md = build_markdown(report, names, dist)
    md_path = os.path.join(OUT_DIR, "plantseg_34_audit.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[OK] MD    -> {md_path}")

    # ---------- 控制台摘要 ----------
    print("\n==== 摘要 ====")
    print(f"数据集: {DATASET_DIR}")
    print(f"34 类全覆盖(train/val/test): {report['all_classes_present']}")
    if not report["all_classes_present"]:
        for sp in SPLITS:
            if report["missing_classes"][sp]:
                print(f"  ❌ {sp} 缺失: {[c['name'] for c in report['missing_classes'][sp]]}")
    print(f"图片总数: train={split_totals['train']['n_images']} "
          f"val={split_totals['val']['n_images']} test={split_totals['test']['n_images']} "
          f"(合计 {total_imgs_all})")
    print(f"AL 子集 al34_random: {al_stat['n_images']} 图, 占 train {al_stat['proportion_of_train']*100:.1f}%")
    print(f"与 val/test 重叠: {'无' if report['al_overlap']['no_overlap_with_val_test'] else '有!'}")
    print(f"未标注池 U0 = {report['unlabeled_pool_U0']['size']}")
    ib = imbalance["by_total_instance"]
    print(f"类别不平衡: max/min 实例比={ib['max_min_ratio']}, CV={ib['cv']}, "
          f"极度不平衡={imbalance['extreme_imbalance_flag']}")
    if rare_inst:
        print(f"  稀有类(<30实例): {rare_inst}")


def build_markdown(report, names, dist):
    L = []
    L.append("# Task-002A · PlantSeg 34 类数据正式审计\n")
    L.append("> 约束：仅读取，未修改任何数据 / 标签 / 划分，未下载外部数据。所有统计来自真实文件。\n")
    L.append("\n## 0. 实际路径记录\n")
    for k, v in report["paths"].items():
        L.append(f"- **{k}**: `{v}`\n")
    L.append(f"- 数据根目录: `{report['paths']['data_root']}`\n")

    L.append("\n## 1. 配置文件检查\n")
    cj = report["config_check"]["classes_json"]
    dy = report["config_check"]["data_yaml"]
    L.append(f"- `classes.json`: 存在={cj['exists']}, 类别数={cj.get('n_classes')}, "
             f"mode={cj.get('mode')}, source={cj.get('source')}\n")
    L.append(f"- `data.yaml`: 存在={dy['exists']}, "
             f"nc={dy.get('nc')}, names 数={dy.get('names_count')}, "
             f"names 与 classes.json 一致={dy.get('names_match_classes_json')}\n")
    L.append(f"  - path 字段: `{dy.get('path_field')}`\n")
    L.append(f"  - note: {cj.get('note')}\n")

    L.append("\n## 2&5. 划分图片总数\n")
    L.append("| 划分 | 图片数 | 标签数 |\n|---|---|---|")
    for sp in SPLITS:
        t = report["split_totals"][sp]
        L.append(f"| {sp} | {t['n_images']} | {t['n_labels']} |\n")
    L.append(f"| **合计** | **{report['total_images']}** | |\n")
    L.append(f"- 空标签图片数(有图无实例): {report['empty_label_images']}\n")

    L.append("\n## 3. 34 类全覆盖确认\n")
    L.append(f"- **结论**: {'✅ 34 类在 train/val/test 均存在有效样本' if report['all_classes_present'] else '❌ 存在缺失类别'}\n")
    for sp in SPLITS:
        miss = report["missing_classes"][sp]
        if miss:
            L.append(f"- ❌ {sp} 缺失类: {[c['name'] for c in miss]}\n")
        else:
            L.append(f"- ✅ {sp}: 34/34 类均有样本\n")

    L.append("\n## 4. 逐类别分布（图片数 / 实例数）\n")
    L.append("| id | 类别 | train图 | train实例 | val图 | val实例 | test图 | test实例 | 总图 | 总实例 |\n")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for d in dist:
        L.append(f"| {d['class_id']} | {d['class_name']} | {d['train_imgs']} | {d['train_inst']} | "
                 f"{d['val_imgs']} | {d['val_inst']} | {d['test_imgs']} | {d['test_inst']} | "
                 f"{d['total_imgs']} | {d['total_inst']} |\n")

    L.append("\n## 6. AL 初始标注集占比 (al34_random)\n")
    a = report["al_subset"]
    L.append(f"- AL 子集图片数: {a['n_images']}，标签数: {a['n_labels']}\n")
    L.append(f"- `dataset_plantseg_34/train` 总数: {a['train_total']}\n")
    L.append(f"- **占比**: {a['proportion_of_train']*100:.2f}%\n")
    L.append(f"- 是否为 train 子集: {a['is_subset_of_train']}（不在 train 中: {len(a['not_in_train'])} 个）\n")
    L.append(f"- `labeled.txt` 去重行数: {a['labeled_txt_unique']}；图不在列表: {len(a['in_al_not_in_labeled'])}；列表无图: {len(a['in_labeled_not_in_al'])}\n")

    L.append("\n## 7. AL 与 val/test 重叠\n")
    o = report["al_overlap"]
    L.append(f"- 与 val 重叠: {len(o['with_val'])} 个\n")
    L.append(f"- 与 test 重叠: {len(o['with_test'])} 个\n")
    L.append(f"- **结论**: {'✅ 完全不重叠' if o['no_overlap_with_val_test'] else '❌ 存在重叠'}\n")

    L.append("\n## 8. 未标注池 U0\n")
    u = report["unlabeled_pool_U0"]
    L.append(f"- {u['formula']}\n")
    L.append(f"- **U0 规模 = {u['size']}**\n")

    L.append("\n## 9. 类别不平衡诊断\n")
    ib = report["class_imbalance"]
    bi = ib["by_total_instance"]
    bt = ib["by_train_instance"]
    L.append("**按总实例数**\n")
    L.append(f"- min={bi['min']}, max={bi['max']}, median={bi['median']}, mean={bi['mean']}, "
             f"std={bi['std']}, CV={bi['cv']}, max/min 比={bi['max_min_ratio']}\n")
    L.append("**按 train 实例数**\n")
    L.append(f"- min={bt['min']}, max={bt['max']}, median={bt['median']}, mean={bt['mean']}, "
             f"std={bt['std']}, CV={bt['cv']}, max/min 比={bt['max_min_ratio']}\n")
    L.append(f"- **极度不平衡标记 (max/min>10)**: {ib['extreme_imbalance_flag']}\n")
    L.append(f"- 稀有类(总实例<30): {ib['rare_classes_total_inst_lt_30'] or '无'}\n")
    L.append(f"- 极低占比类(占比<0.5%): {ib['very_low_share_lt_0.5pct'] or '无'}\n")

    L.append("\n## 修改文件列表\n")
    L.append("- 仅生成以下审计产物（未修改任何原始数据 / 标签 / 配置）：\n")
    L.append(f"  - `{os.path.join(OUT_DIR, 'plantseg_34_audit.md')}`\n")
    L.append(f"  - `{os.path.join(OUT_DIR, 'plantseg_34_audit.json')}`\n")
    L.append(f"  - `{os.path.join(OUT_DIR, 'class_distribution.csv')}`\n")
    return "".join(L)


if __name__ == "__main__":
    main()
