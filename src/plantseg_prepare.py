"""把 PlantSeg 整理成 YOLO 分割格式。

解压后结构（约 8 GB）：

    plantseg/
      images/{train,val,test}/       7,774 张自然场景图（5,367 / 846 / 1,561）
      annotations/                   7,774 个单通道 PNG，像素值 = 组合编号
      annotation_{train,val,test}.json   COCO 格式，但**不可用**，见下
      Metadata.csv                   Name,Index,Plant,Disease,Resolution,
                                     Label file,Mask ratio,URL,License,Split

## 掩码编码：像素值 = Index + 1

逐类抽样 115 个 Index 全部验证通过，偏移恒为 1，没有例外。
原因：PIL 保存 8 位调色板 PNG 时背景必须占用 0 号槽位，
作者把类别表整体后移了一格。所以像素值不能直接当类别号用，
必须减 1 才是 Metadata 的 Index。漏了这一步，115 个类会整体错位，
且第 0 类永远学不到（它的掩码值是 1，会被误认为第 1 类）。

## 为什么不用官方 COCO JSON

实测三份 JSON 无法对齐，按 image_id 匹配会**给验证/测试图配上训练图的病斑坐标**：

  - image_id 在每个划分里独立编号：train 1~7915、val 1~1247、test 1~2295
  - 同一个 id 指向不同文件：id=1 在 train 是 apple_black_rot_1.jpg，
    在 val 是 apple_black_rot_28.jpg
  - annotation_train.json 的 annotations 是全局并集（train∩val 恰好
    等于 val 的全部 8,927 条），但它的 images 只有 train 自己的 5,367 条

后果：train 的 52,898 条标注里只有 38,251 条能对上自己的图；
另外 79,916 个唯一 id 中有 2,549 个在 train 的 images 里找不到。

所以改用 **PNG 掩码 + Metadata.csv** 这条唯一自洽的路：
掩码与图片同名一一对应，划分和类别都从 Metadata.csv 取。

代价：同一片叶子上挨得极近的病斑会被连通域粘成一个实例。
官方 COCO 里是分开的，但那份数据我们不敢用。这个代价要写进报告。

## 类别空间

Metadata 的 `Index`（0~114）是「作物-病害组合」编号。
注意 `Disease` 字段本身已带作物名（如 apple black rot），
所以「只按病种」和「按组合」是同一套 115 类，没有 69 类这回事。
真正有意义的是两档：combo（115）和 host（34）。

用法：
  python src/plantseg_prepare.py --inspect
  python src/plantseg_prepare.py --class-mode combo --out-root H:/Plant/_derived
"""
import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from common import setup_matplotlib  # noqa: F401  （保持与其他脚本一致的中文绘图设置）

RAW = Path("H:/Plant/_raw/plantseg/plantseg")
SPLIT_DIR = {"training": "train", "validation": "val", "valid": "val", "test": "test"}


def imread_u(path, flags=cv2.IMREAD_COLOR):
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def mask_instances(mask, min_area, max_points, eps_ratio):
    """单通道掩码 -> [(掩码值, 归一化多边形)]，每个连通域一个实例。"""
    h, w = mask.shape[:2]
    out = []
    for val in np.unique(mask):
        v = int(val)
        if v == 0:
            continue
        binm = (mask == v).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < min_area:
                continue
            comp = ((lab == k) * 255).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) < min_area:
                continue
            approx = cv2.approxPolyDP(cnt, eps_ratio * cv2.arcLength(cnt, True), True)
            approx = approx.reshape(-1, 2)
            if len(approx) < 3:
                continue
            if len(approx) > max_points:
                step = len(approx) / max_points
                approx = np.array([approx[int(i * step)] for i in range(max_points)])
            pts = np.stack([approx[:, 0] / w, approx[:, 1] / h], axis=1)
            out.append((v, np.clip(pts, 0.0, 1.0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--class-mode", default="combo", choices=["combo", "host"])
    ap.add_argument("--out-name", default="dataset_plantseg")
    ap.add_argument("--out-root", default=None,
                    help="输出根目录；放在与原始数据同卷上才能硬链接，省掉 8 GB 复制")
    ap.add_argument("--min-area", type=int, default=16)
    ap.add_argument("--max-points", type=int, default=120)
    ap.add_argument("--eps-ratio", type=float, default=0.004)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.raw)
    meta_path = root / "Metadata.csv"
    if not meta_path.exists():
        raise SystemExit(f"{root} 下没有 Metadata.csv，路径对吗？")

    with open(meta_path, encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        metas = list(rd)
    head = list(rd.fieldnames or [])

    idx_of_pair = {}
    for r in metas:
        i = int(r["Index"])
        pair = (r["Plant"].strip(), r["Disease"].strip())
        idx_of_pair.setdefault(i, pair)
    hosts = sorted({p for p, _ in idx_of_pair.values()})

    if args.inspect:
        print(f"根目录：{root}")
        print(f"Metadata 表头：{head}")
        print(f"行数 {len(metas)}")
        print("Split 取值：", dict(Counter(r["Split"].strip() for r in metas)))
        print(f"作物-病害组合 {len(idx_of_pair)} | 宿主 {len(hosts)}")
        print(f"宿主：{hosts}")
        samp = metas[0]
        sp0 = SPLIT_DIR.get(samp["Split"].strip().lower(), "train")
        p = root / "annotations" / sp0 / samp["Label file"].strip()
        m = imread_u(p, cv2.IMREAD_UNCHANGED)
        vals = sorted(int(v) for v in np.unique(m))
        print(f"样例掩码 {p.name}: shape={m.shape} dtype={m.dtype} "
              f"取值 {vals[:8]} 前景占比 {(m > 0).mean():.3f}")
        return

    # ------------------------------------------------ 类别表
    if args.class_mode == "combo":
        raw2name = {i: f"{p}___{d}" for i, (p, d) in idx_of_pair.items()}
    else:
        raw2name = {i: p for i, (p, _) in idx_of_pair.items()}
    uniq = sorted(set(raw2name.values()))
    cls_of = {n: i for i, n in enumerate(uniq)}
    raw2yolo = {raw: cls_of[n] for raw, n in raw2name.items()}
    print(f"类别空间（{args.class_mode}）：{len(uniq)} 类")

    out = (Path(args.out_root) if args.out_root else PROJECT / "out") / args.out_name
    for sub in ["images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    with open(out / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": uniq, "mode": args.class_mode,
                   "source": "PlantSeg Metadata.csv (Plant/Disease/Index)",
                   "note": "官方 COCO JSON 三划分 image_id 不唯一，"
                           "实例由 PNG 掩码连通域重建"},
                  f, ensure_ascii=False, indent=2)

    rows = metas[: args.limit] if args.limit else metas
    drop = Counter()
    per_split_n = Counter()
    per_class_img = Counter()
    n_inst_total = 0
    offset_bad = 0
    idx_skew = 0

    for n, r in enumerate(rows, 1):
        name = r["Name"].strip()
        lab = r["Label file"].strip()
        sp = SPLIT_DIR.get(r["Split"].strip().lower())
        if sp is None:
            drop["划分未知"] += 1
            continue
        img_p = root / "images" / sp / name
        if not img_p.exists():
            hits = list((root / "images").rglob(name))
            if not hits:
                drop["找不到图"] += 1
                continue
            img_p = hits[0]
        mask_p = root / "annotations" / sp / lab
        if not mask_p.exists():
            hits = list((root / "annotations").rglob(lab))
            if not hits:
                drop["找不到掩码"] += 1
                continue
            mask_p = hits[0]

        mask = imread_u(mask_p, cv2.IMREAD_UNCHANGED)
        if mask is None:
            drop["掩码读失败"] += 1
            continue
        if mask.ndim == 3:
            mask = mask[..., 0]

        meta_cid = raw2yolo.get(int(r["Index"]))
        if meta_cid is None:
            drop["类别不在表里"] += 1
            continue

        insts = mask_instances(mask, args.min_area, args.max_points, args.eps_ratio)
        if not insts:
            drop["无有效病斑"] += 1
            continue

        # 自检：掩码的原始取值应当恰好是 {Index + 1}。
        # 实测 7,773 张里有 12 张对不上（偏差 +2，但作物/病害名是对的），
        # 是作者用旧版类别表做的图。这类图一律以 Metadata 为准，
        # 否则按「掩码值减 1」反解会整体错到一个邻居类上。
        raw_vals = {int(v) for v in np.unique(mask) if v != 0}
        trust_mask = bool(raw_vals) and raw_vals == {int(r["Index"]) + 1}
        if not trust_mask:
            idx_skew += 1

        lines = []
        for v, pts in insts:
            if trust_mask:
                # 掩码像素值 = Index + 1（见文件头说明），减 1 换回 Index
                cid = raw2yolo.get(v - 1, meta_cid)
            else:
                cid = meta_cid
            if cid is None:
                offset_bad += 1
                cid = meta_cid
            lines.append(str(cid) + " " +
                         " ".join(f"{x:.6f} {y:.6f}" for x, y in pts))

        dst_img = out / "images" / sp / name
        dst_lab = out / "labels" / sp / (Path(name).stem + ".txt")
        try:
            dst_img.hardlink_to(img_p)
        except (OSError, AttributeError):
            if not dst_img.exists():
                shutil.copy2(img_p, dst_img)
        dst_lab.write_text("\n".join(lines), encoding="utf-8")

        per_split_n[sp] += 1
        n_inst_total += len(lines)
        per_class_img[cls_of[raw2name[int(r["Index"])]]] += 1

        if n % 500 == 0:
            print(f"  {n}/{len(rows)} ... 实例 {n_inst_total}")

    yaml_path = out / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {out.as_posix()}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write(f"nc: {len(uniq)}\nnames:\n")
        for i, c in enumerate(uniq):
            f.write(f"  {i}: {c}\n")

    cnt = np.array(sorted(per_class_img.values(), reverse=True))
    print(f"\n合计 {sum(per_split_n.values())} 图 / {n_inst_total} 实例 / {len(uniq)} 类")
    for sp in ["train", "val", "test"]:
        print(f"  {sp}: {per_split_n[sp]} 图")
    for k, v in drop.most_common():
        print(f"  丢弃 {k}: {v}")
    print(f"  掩码取值与 Index+1 不一致的图: {idx_skew}")
    print(f"  掩码取值越界（退回元数据类别）的实例: {offset_bad}")
    if cnt.size:
        print(f"每类图像数：最大 {cnt.max()} 中位 {int(np.median(cnt))} "
              f"最小 {cnt.min()} 长尾比 {cnt.max()/max(cnt.min(),1):.1f}:1")
    print(f"配置：{yaml_path}")


if __name__ == "__main__":
    main()
