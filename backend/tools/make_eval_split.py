"""按 id 列表切出一个评测子集，供 m6_editcost 只评这些图。

## 为什么需要它

`m6_editcost.load_yolo_seg` 把「标签文件不存在」视为「这张图没有目标」。
三臂实验只标 `al_pool/unlabeled_pool.txt` 里的 36 张（全部在 train），
若直接 `--split train`，剩下 30 张未标会被当成"整张误检"，
`标为无目标` 计数暴涨，三个臂的编辑代价全部失真。

所以先切一个只含这 36 张的子集，再对它跑 m6：

    python backend/tools/make_eval_split.py \
        --ids datasets/plantseg100_pack/al_pool/unlabeled_pool.txt \
        --out .tmp/eval_split

    python src/m6_editcost.py \
        --data .tmp/eval_split/data.yaml --split train \
        --weights weights/baseline_yolov8nseg_plantseg34_best.pt \
        --labels-dir out/annotations/<arm> --name <arm>

子集里是**复制**的图（不是软链，Windows 软链要管理员权限），
36 张约 6 MB，可以随时删了重生成。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
PACK = PROJECT / "datasets" / "plantseg100_pack"


def read_ids(path: Path) -> list[str]:
    """接受三种写法：文件名.jpg / 文件名 / image_id。统一成 stem。"""
    stems: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        stems.append(Path(s).stem)
    return list(dict.fromkeys(stems))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(PACK), help="采样包根目录")
    ap.add_argument("--ids", default=None,
                    help="id 列表文件；缺省用包里的 al_pool/unlabeled_pool.txt")
    ap.add_argument("--split", default="train", help="源划分（默认 train）")
    ap.add_argument("--out", default=str(PROJECT / ".tmp" / "eval_split"))
    args = ap.parse_args()

    pack = Path(args.pack)
    ids_file = (Path(args.ids) if args.ids
                else pack / "al_pool" / "unlabeled_pool.txt")
    if not ids_file.is_file():
        raise SystemExit(f"找不到 id 列表：{ids_file}")

    src_dir = pack / "images" / args.split
    if not src_dir.is_dir():
        raise SystemExit(f"找不到划分目录：{src_dir}")

    by_stem = {p.stem: p for p in src_dir.iterdir() if p.is_file()}
    stems = read_ids(ids_file)
    missing = [s for s in stems if s not in by_stem]
    if missing:
        print(f"  ⚠ {len(missing)} 个 id 不在 {args.split}/ 里，已跳过：{missing[:5]}")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True)

    copied = 0
    for s in stems:
        src = by_stem.get(s)
        if src is None:
            continue
        shutil.copy2(src, img_dir / src.name)
        copied += 1

    # data.yaml：path 相对本文件（m6 已按 yaml 所在目录解析）
    names = json.loads((pack / "classes.json").read_text(encoding="utf-8"))["classes"]
    lines = [
        f"# 由 backend/tools/make_eval_split.py 生成，勿手改",
        f"# ids: {ids_file.name}  split: {args.split}  张数: {copied}",
        "path: .",
        "train: images",
        "val: images",
        f"nc: {len(names)}",
        "names:",
    ]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"  ✔ {copied} 张 -> {out / 'images'}")
    print(f"  ✔ 配置 -> {out / 'data.yaml'}")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
