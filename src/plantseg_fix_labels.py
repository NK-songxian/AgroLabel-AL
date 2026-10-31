"""定点修复「掩码值不等于 Index+1」的那十几张图的标签。

背景见 plantseg_prepare.py 文件头：PlantSeg 的 PNG 掩码像素值本应是
`Index + 1`，但全库 7,773 张里有 12 张用的是旧版类别表，偏差 +2。
这些图的作物、病害在 Metadata 里是对的，只有掩码值错位，
所以正确的做法是**以 Metadata 为准**重写它们的 YOLO 标签。

为什么要单独一个脚本：全量重跑一次转换要 20 多分钟（8 GB 掩码要逐张解码），
而真正需要修的只有 12 张。这个脚本只重新解码这些图，几秒跑完。

用法：
  python src/plantseg_fix_labels.py --derived H:/Plant/_derived
  python src/plantseg_fix_labels.py --derived H:/Plant/_derived --dry-run
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from plantseg_prepare import SPLIT_DIR, mask_instances  # noqa: E402

RAW = Path("H:/Plant/_raw/plantseg/plantseg")


def imread_u(path, flags=cv2.IMREAD_COLOR):
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--derived", default="H:/Plant/_derived")
    ap.add_argument("--min-area", type=int, default=16)
    ap.add_argument("--max-points", type=int, default=120)
    ap.add_argument("--eps-ratio", type=float, default=0.004)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.raw)
    with open(root / "Metadata.csv", encoding="utf-8-sig") as f:
        metas = list(csv.DictReader(f))

    # 先扫一遍，挑出掩码值与 Index+1 不一致的图
    skew = []
    for r in metas:
        sp = SPLIT_DIR.get(r["Split"].strip().lower())
        if sp is None:
            continue
        m = imread_u(root / "annotations" / sp / r["Label file"].strip(),
                     cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            m = m[..., 0]
        vals = {int(v) for v in np.unique(m) if v != 0}
        if not vals:
            continue
        if vals != {int(r["Index"]) + 1}:
            skew.append((r, sp, m, sorted(vals)))

    print(f"掩码取值与 Index+1 不一致的图：{len(skew)} 张")
    if not skew:
        return

    # 对每个派生数据集分别修
    for ds in sorted(p for p in Path(args.derived).iterdir()
                     if p.is_dir() and (p / "classes.json").exists()):
        cls_path = ds / "classes.json"
        info = json.loads(cls_path.read_text(encoding="utf-8"))
        classes = info["classes"]
        mode = info.get("mode", "combo")
        print(f"\n[{ds.name}] {len(classes)} 类 mode={mode}")

        fixed = 0
        for r, sp, m, vals in skew:
            plant, dis = r["Plant"].strip(), r["Disease"].strip()
            name = f"{plant}___{dis}" if mode == "combo" else plant
            if name not in classes:
                print(f"  跳过（类名不在表里）：{name}")
                continue
            cid = classes.index(name)

            stem = Path(r["Name"].strip()).stem
            lab = ds / "labels" / sp / f"{stem}.txt"
            if not lab.exists():
                print(f"  跳过（无标签文件）：{lab}")
                continue
            old = sorted({int(l.split()[0])
                          for l in lab.read_text(encoding="utf-8").splitlines()})
            if old == [cid]:
                continue
            if args.dry_run:
                print(f"  {stem:40s} {old} -> [{cid}]  ({name})")
                continue
            # 几何不变，只换类别号：直接改写每行的首列
            lines = []
            for l in lab.read_text(encoding="utf-8").splitlines():
                p = l.split()
                if len(p) < 7:
                    continue
                lines.append(str(cid) + " " + " ".join(p[1:]))
            lab.write_text("\n".join(lines), encoding="utf-8")
            print(f"  {stem:40s} {old} -> [{cid}]  ({name})")
            fixed += 1

        if not args.dry_run:
            print(f"  修复 {fixed} 个标签文件")


if __name__ == "__main__":
    main()
