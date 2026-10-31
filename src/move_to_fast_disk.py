"""把 PlantSeg 派生数据集从 H: 搬到 F:，让训练时的 IO 不再是瓶颈。

## 为什么要搬

实测这台机器上 H: 盘读只有 0.2~10 MB/s、写 11 MB/s，而 F: 写 124 MB/s。
YOLO 每建一次图像缓存要写 7 GB，放在 H: 上光是写缓存就要二十分钟以上，
而且主动学习要反复训练二十来次，IO 会一直卡在那。

## 怎么搬才省时间

`dataset_plantseg_34` 和 `dataset_plantseg_115` 用的是**同一批图片**，
只有标签不同。所以图片只拷一份（放在 34 那份目录里），
115 那份的 images 用硬链接指过去——同一块盘上硬链接不占额外空间，
Ultralytics 按路径读图，看不出区别。标签本身只有几十 MB，各拷一份。

用法：
  python src/move_to_fast_disk.py --src H:/Plant/_derived --dst F:/_plantseg
  python src/move_to_fast_disk.py --src H:/Plant/_derived --dst F:/_plantseg --only 34
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

SPLITS = ["train", "val", "test"]


def copy_images(src_root, dst_root, splits):
    total = sum(len(list((src_root / "images" / s).glob("*.*"))) for s in splits)
    done = 0
    t0 = time.time()
    for s in splits:
        d = dst_root / "images" / s
        d.mkdir(parents=True, exist_ok=True)
        srcs = [p for p in (src_root / "images" / s).iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        for p in srcs:
            dpath = d / p.name
            if not dpath.exists():
                shutil.copy2(p, dpath)
            done += 1
            if done % 500 == 0:
                el = time.time() - t0
                print(f"    图 {done}/{total}  {done/el:.0f} 张/秒  "
                      f"剩 {el/done*(total-done)/60:.1f} 分钟")
    return done


def copy_labels(src_root, dst_root, splits):
    n = 0
    for s in splits:
        d = dst_root / "labels" / s
        d.mkdir(parents=True, exist_ok=True)
        for p in (src_root / "labels" / s).glob("*.txt"):
            dp = d / p.name
            if not dp.exists():
                shutil.copy2(p, dp)
            n += 1
    return n


def link_images(src_root, dst_root, splits):
    n = 0
    skip = 0
    for s in splits:
        d = dst_root / "images" / s
        d.mkdir(parents=True, exist_ok=True)
        for p in (src_root / "images" / s).iterdir():
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            dp = d / p.name
            if dp.exists():
                skip += 1
                continue
            try:
                dp.hardlink_to(p)
                n += 1
            except OSError:
                shutil.copy2(p, dp)
                n += 1
    return n, skip


def write_yaml(dst_root):
    src_yaml = dst_root / "data_from_h.yaml"
    names, nc = {}, 0
    if src_yaml.exists():
        import yaml
        d = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
        names, nc = d.get("names") or {}, d.get("nc") or 0
    y = dst_root / "data.yaml"
    with open(y, "w", encoding="utf-8") as f:
        f.write(f"path: {dst_root.as_posix()}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write(f"nc: {nc}\nnames:\n")
        for k, v in names.items():
            f.write(f"  {k}: {v}\n")
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="H:/Plant/_derived")
    ap.add_argument("--dst", default="F:/_plantseg")
    ap.add_argument("--only", nargs="*", default=["34", "115"],
                    help="只搬哪几个类别空间")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    primary = None
    for tag in args.only:
        name = f"dataset_plantseg_{tag}"
        s_root = src / name
        if not s_root.exists():
            print(f"跳过 {name}（不存在）")
            continue
        d_root = dst / name
        print(f"\n[{tag} 类] {s_root} -> {d_root}")

        if primary is None:
            t = time.time()
            n = copy_images(s_root, d_root, SPLITS)
            print(f"  复制图片 {n} 张，{time.time()-t:.0f} 秒")
            primary = d_root
        else:
            a, b = link_images(primary, d_root, SPLITS)
            print(f"  图片硬链接 {a} 张（已有 {b} 张）")

        n = copy_labels(s_root, d_root, SPLITS)
        print(f"  标签 {n} 个")
        shutil.copy2(s_root / "classes.json", d_root / "classes.json")
        shutil.copy2(s_root / "data.yaml", d_root / "data_from_h.yaml")
        y = write_yaml(d_root)
        print(f"  配置 {y}")

    # 顺手把从 H: 复制过来的路径在 data_from_h.yaml 里保留，便于回溯
    print("\n完成。训练时把 --data 指到 F: 下的 data.yaml 即可。")


if __name__ == "__main__":
    main()
