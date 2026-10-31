"""M3 第一步：把伪标签整理成 ultralytics 能直接吃的目录结构。

ultralytics 认的是这套约定：
  dataset/
    images/train/xxx.jpg
    labels/train/xxx.txt     ← 跟图片同名，只把 images 换成 labels
    images/val/ ...
    data.yaml
标签必须和图片放在「对应的两个目录」里，靠路径替换去找，
所以没法只给一个文件清单了事，图片得实际落到新目录下。

运行：
  python src/m3_dataset.py
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config

PROJECT = Path(__file__).resolve().parents[1]


def link_or_copy(src: Path, dst: Path):
    """优先硬链接（不占额外空间），跨盘或失败就退回复制。"""
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except (OSError, AttributeError):
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT / "configs" / "default.yaml"))
    ap.add_argument("--val-ratio", type=float, default=0.12,
                    help="从伪标注池里划多少比例做验证集")
    args = ap.parse_args()

    cfg = load_config(args.config)
    index_path = PROJECT / cfg["paths"]["pseudo"] / "index.csv"
    if not index_path.exists():
        raise SystemExit("先跑 m2_pseudo.py 生成 out/pseudo/index.csv")

    with open(PROJECT / cfg["paths"]["classes_json"], encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    rows = []
    with open(index_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["ok"] == "1" and r["label"] and Path(r["label"]).exists():
                rows.append(r)
    if not rows:
        raise SystemExit("index.csv 里没有合格样本")

    ds = PROJECT / "out" / "dataset"
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (ds / sub).mkdir(parents=True, exist_ok=True)

    # 按路径哈希切分，不按顺序切：原始目录是按类排的，
    # 顺序切会让验证集里缺掉字母序靠后的类。
    n_val = 0
    for r in rows:
        h = int(hashlib.md5(r["image"].encode("utf-8")).hexdigest()[:8], 16)
        part = "val" if (h % 1000) < args.val_ratio * 1000 else "train"

        img_src = Path(r["image"])
        img_dst = ds / "images" / part / img_src.name
        lab_dst = ds / "labels" / part / (img_src.stem + ".txt")
        # 不同类别目录下可能有同名文件（image (1).JPG），撞名就加前缀
        if img_dst.exists():
            key = hashlib.md5(r["image"].encode("utf-8")).hexdigest()[:6]
            img_dst = ds / "images" / part / f"{img_src.stem}_{key}{img_src.suffix}"
            lab_dst = ds / "labels" / part / f"{img_src.stem}_{key}.txt"

        link_or_copy(img_src, img_dst)
        shutil.copy2(r["label"], lab_dst)
        if part == "val":
            n_val += 1

    yaml_path = ds / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {ds.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(classes)}\n")
        f.write("names:\n")
        for i, c in enumerate(classes):
            f.write(f"  {i}: {c}\n")

    print("数据集就绪：", ds)
    print(f"  train {len(rows) - n_val} 张 / val {n_val} 张 / {len(classes)} 类")
    print(f"  配置：{yaml_path}")


if __name__ == "__main__":
    main()
