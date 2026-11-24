"""生成「同图三臂 × 多组」实验任务清单。

设计（为什么这样抽）
--------------------
- **同图三臂**：同一张图在 assisted / manual / noprompt 各标一遍。
  图难度被完全消掉，每张图给出一个配对差 d = cost(臂A) − cost(臂B)。
- **多组**：把图分成 G 组，每组 x 张。组用于估计"换一批图结论还成立吗"。
  标注量 = x × 3 × G。
- **臂顺序轮转**：同一张图的三个任务按 (组号 + 组内序号) 旋转起始臂，
  使三个臂在"第 1/2/3 遍"上分布均衡 —— 否则"第三遍最快"会被误读成某个臂更好。
- **随机交错**（默认）：把所有任务打散。记忆效应与顺序效应都被打散，
  而不是集中在某一段。

用法
----
    python backend/tools/make_experiment_manifest.py \
        --data "C:/Users/1/Desktop/附件/AL_v4/dataset/data.yaml" \
        --x 8 --groups 4 --seed 20261124 \
        --out out/experiment/manifest.json

    # 按组内顺序执行（不想被随机跳转打扰时用）
    ... --layout by-group
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ARMS = ("assisted", "manual", "noprompt")
PROJECT = Path(__file__).resolve().parents[2]


def load_names(data_yaml: Path) -> list[str]:
    """读 data.yaml 的 names（只用来给清单补 class_name，纯展示用）。"""
    try:
        import yaml
    except ImportError:
        return []
    try:
        d = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except OSError:
        return []
    names = d.get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(x) for x in names]


def class_of(image_id: str, class_names: list[str], labels_dir: Path | None) -> str | None:
    """从 labels/<split>/<stem>.txt 的第一行取类别 id，映射成类名。"""
    if labels_dir is None or not class_names:
        return None
    lbl = labels_dir / f"{image_id}.txt"
    if not lbl.is_file():
        return None
    try:
        first = lbl.read_text(encoding="utf-8").splitlines()[0]
        cid = int(first.split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return class_names[cid] if 0 <= cid < len(class_names) else None


def main() -> int:
    ap = argparse.ArgumentParser(description="生成同图三臂实验任务清单")
    ap.add_argument("--data", required=True, help="data.yaml 路径（用来定位图片与类名）")
    ap.add_argument("--split", default="val", help="从哪个 split 抽图（默认 val）")
    ap.add_argument("--x", type=int, default=8, help="每组抽多少张图")
    ap.add_argument("--groups", type=int, default=4, help="做几组")
    ap.add_argument("--seed", type=int, default=20261124, help="随机种子（可复现）")
    ap.add_argument("--layout", choices=("interleave", "by-group"), default="interleave",
                    help="interleave=所有任务打散（推荐）；by-group=组内打散、组间顺序")
    ap.add_argument("--out", default=str(PROJECT / "out" / "experiment" / "manifest.json"))
    ap.add_argument("--name", default=None, help="清单名（默认自动生成）")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="排除这些 image_id（例如已经标过的）")
    args = ap.parse_args()

    if args.x < 1 or args.groups < 1:
        print("--x 和 --groups 必须 >= 1", file=sys.stderr)
        return 2

    data_yaml = Path(args.data).expanduser()
    if not data_yaml.is_file():
        print(f"找不到 data.yaml: {data_yaml}", file=sys.stderr)
        return 2

    root = data_yaml.parent
    try:
        import yaml  # noqa: F401
        conf = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except Exception:
        conf = {}
    declared = Path(str(conf.get("path") or root))
    dataset_root = declared if declared.is_dir() else root

    img_dir = dataset_root / "images" / args.split
    lbl_dir = dataset_root / "labels" / args.split
    if not img_dir.is_dir():
        print(f"找不到图片目录: {img_dir}", file=sys.stderr)
        return 2

    pool = sorted(p.stem for p in img_dir.glob("*.jpg"))
    skip = set(args.exclude)
    pool = [i for i in pool if i not in skip]
    need = args.x * args.groups
    if len(pool) < need:
        print(f"可用图 {len(pool)} 张，不够 {need} 张（x={args.x} × groups={args.groups}）",
              file=sys.stderr)
        return 2

    rnd = random.Random(args.seed)
    picked = rnd.sample(pool, need)          # 不放回：组间也不重图
    names = load_names(data_yaml)

    groups: list[dict] = []
    tasks: list[dict] = []
    for gi in range(args.groups):
        gid = f"g{gi + 1}"
        members = picked[gi * args.x:(gi + 1) * args.x]
        groups.append({"id": gid, "images": members})
        for j, iid in enumerate(members):
            # 起始臂按 (组号 + 组内序号) 旋转，保证三个臂在 3 个位置上均衡
            start = (gi + j) % 3
            order = [ARMS[(start + k) % 3] for k in range(3)]
            for arm in order:
                tasks.append({
                    "image_id": iid,
                    "arm": arm,
                    "group": gid,
                    "class_name": class_of(iid, names, lbl_dir),
                })

    # 执行顺序
    idx = list(range(len(tasks)))
    if args.layout == "by-group":
        # 组内打散，组间按 g1,g2,...
        ordered: list[dict] = []
        for gi in range(args.groups):
            block = [t for t in tasks if t["group"] == f"g{gi + 1}"]
            rnd.shuffle(block)
            ordered.extend(block)
    else:
        rnd.shuffle(idx)
        ordered = [tasks[i] for i in idx]
    for n, t in enumerate(ordered, start=1):
        t["order"] = n

    manifest = {
        "name": args.name or f"同图三臂 · {args.x}图×{args.groups}组",
        "design": {
            "kind": "same-image-3arms",
            "x": args.x,
            "groups": args.groups,
            "arms": list(ARMS),
            "split": args.split,
            "seed": args.seed,
            "layout": args.layout,
            "total_tasks": len(ordered),
            "note": "同一张图在三个臂各标一遍；臂顺序按(组号+组内序号)轮转",
        },
        "groups": groups,
        "tasks": ordered,
    }

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 自检输出
    per_arm = Counter(t["arm"] for t in ordered)
    per_pos = Counter()
    for gi in range(args.groups):
        for j in range(args.x):
            start = (gi + j) % 3
            for pos in range(3):
                per_pos[f"第{pos + 1}遍:{ARMS[(start + pos) % 3]}"] += 1
    cls_img = {}
    for t in ordered:
        cls_img.setdefault(t["image_id"], t["class_name"] or "（无标签）")

    print(f"  清单 -> {out}")
    print(f"  {args.x} 图 × 3 臂 × {args.groups} 组 = {len(ordered)} 个任务"
          f"（{len(set(t['image_id'] for t in ordered))} 张图）")
    print(f"  每臂任务数: {dict(sorted(per_arm.items()))}")
    print(f"  位置×臂 分布: {dict(sorted(per_pos.items()))}")
    print(f"  覆盖类别: {len(set(cls_img.values()))} 个")
    only_once = [c for c, n in Counter(cls_img.values()).items() if n == 1]
    if only_once:
        print(f"  只出现 1 张的类别: {len(only_once)} 个（{', '.join(only_once[:6])}…）")
    print(f"  执行顺序: {args.layout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
