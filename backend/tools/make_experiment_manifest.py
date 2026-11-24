"""生成「同图三臂」实验任务清单。

设计要点
--------
- **同图三臂**：同一张图在 assisted / manual / noprompt 各标一遍。
  图难度被消掉，每张图给出一个配对差 d = cost(臂A) − cost(臂B)。
- **多组**：x 张/组 × G 组。组用于估"换一批图结论还成不成立"。
- **臂优先 + 臂顺序轮转**（默认 layout=by-arm）：
  一组内先把某个臂的图全部做完，再切下一个臂 —— 符合实际标注习惯。
  但"第几遍"会与臂绑定，所以**每组的臂顺序按组号轮转**：
  g1: A→M→N，g2: M→N→A，g3: N→A→M，g4: A→M→N …
  这样三个臂在"第 1/2/3 个做的臂"上分布均衡，块顺序效应可被估计并扣掉。
  每个臂块内部的图顺序**随机打乱**，避免"先做的图更仔细"。
- **多人互不重叠**：`--persons N --person I` 从同一份抽样里切出第 I 个人的
  x×G 张图。两人之间**没有一张重复** —— 因为落盘路径是
  `out/annotations/<arm>/<image_id>.json`，只按臂分目录，
  两个人标同一张同一臂会**互相覆盖**。标注员姓名仍写进 sidecar 的 operator。

用法
----
    # 单人 8 图 × 4 组
    python backend/tools/make_experiment_manifest.py --data .../data.yaml --x 8 --groups 4

    # 两个人，各 8×4，互不重叠
    ... --persons 2 --person 1 --out out/experiment/p1.json
    ... --persons 2 --person 2 --out out/experiment/p2.json

    # 想抗疲劳更强（严格交错，不按臂分块）
    ... --layout interleave
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


def load_conf(data_yaml: Path) -> tuple[Path, list[str]]:
    """返回 (数据集根目录, 类名列表)。类名只用于清单展示。"""
    root = data_yaml.parent
    names: list[str] = []
    declared: Path | None = None
    try:
        import yaml

        conf = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        if conf.get("path"):
            declared = Path(str(conf["path"]))
        n = conf.get("names") or []
        if isinstance(n, dict):
            n = [n[k] for k in sorted(n)]
        names = [str(x) for x in n]
    except Exception:
        pass
    base = declared if (declared and declared.is_dir()) else root
    return base, names


def class_of(image_id: str, class_names: list[str], labels_dir: Path | None) -> str | None:
    if labels_dir is None or not class_names:
        return None
    lbl = labels_dir / f"{image_id}.txt"
    if not lbl.is_file():
        return None
    try:
        cid = int(lbl.read_text(encoding="utf-8").splitlines()[0].split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return class_names[cid] if 0 <= cid < len(class_names) else None


def build_order(tasks: list[dict], layout: str, x: int, groups: int,
                rnd: random.Random) -> list[dict]:
    """按 layout 决定执行顺序，并写 order（1-based）。"""
    if layout == "interleave":
        out = list(tasks)
        rnd.shuffle(out)
    elif layout == "by-group":
        out = []
        for gi in range(groups):
            block = [t for t in tasks if t["group"] == f"g{gi + 1}"]
            rnd.shuffle(block)
            out.extend(block)
    else:  # by-arm：组内按臂分块，臂顺序按组号轮转
        out = []
        for gi in range(groups):
            gid = f"g{gi + 1}"
            for k in range(3):
                arm = ARMS[(gi + k) % 3]
                block = [t for t in tasks if t["group"] == gid and t["arm"] == arm]
                rnd.shuffle(block)          # 块内图序随机
                out.extend(block)
    for n, t in enumerate(out, start=1):
        t["order"] = n
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="生成同图三臂实验任务清单")
    ap.add_argument("--data", required=True, help="data.yaml（用来定位图片与类名）")
    ap.add_argument("--split", default="val", help="从哪个 split 抽图（默认 val）")
    ap.add_argument("--x", type=int, default=8, help="每组抽多少张图")
    ap.add_argument("--groups", type=int, default=4, help="做几组")
    ap.add_argument("--persons", type=int, default=1,
                    help="几个标注员（>1 时每人分到互不重叠的一套图）")
    ap.add_argument("--person", type=int, default=1, help="这是第几个标注员（1-based）")
    ap.add_argument("--seed", type=int, default=20261124, help="随机种子（可复现）")
    ap.add_argument("--layout", choices=("by-arm", "interleave", "by-group"), default="by-arm",
                    help="by-arm=一组内按臂分块（默认，符合标注习惯）；interleave=全随机交错")
    ap.add_argument("--out", default=str(PROJECT / "out" / "experiment" / "manifest.json"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--exclude", nargs="*", default=[], help="排除这些 image_id")
    args = ap.parse_args()

    if args.x < 1 or args.groups < 1 or args.persons < 1:
        print("--x / --groups / --persons 必须 >= 1", file=sys.stderr)
        return 2
    if not 1 <= args.person <= args.persons:
        print(f"--person 必须在 1..{args.persons} 之间", file=sys.stderr)
        return 2

    data_yaml = Path(args.data).expanduser()
    if not data_yaml.is_file():
        print(f"找不到 data.yaml: {data_yaml}", file=sys.stderr)
        return 2

    base, names = load_conf(data_yaml)
    img_dir = base / "images" / args.split
    lbl_dir = base / "labels" / args.split
    if not img_dir.is_dir():
        print(f"找不到图片目录: {img_dir}", file=sys.stderr)
        return 2

    pool = sorted(p.stem for p in img_dir.glob("*.jpg"))
    skip = set(args.exclude)
    pool = [i for i in pool if i not in skip]
    per_person = args.x * args.groups
    need = per_person * args.persons
    if len(pool) < need:
        print(f"可用图 {len(pool)} 张，不够 {need} 张"
              f"（x={args.x} × groups={args.groups} × persons={args.persons}）",
              file=sys.stderr)
        return 2

    rnd = random.Random(args.seed)
    sampled = rnd.sample(pool, need)
    # 每人一段连续切片：同一 seed 下第 i 人拿到的图固定且互不重叠
    picked = sampled[(args.person - 1) * per_person: args.person * per_person]

    groups: list[dict] = []
    tasks: list[dict] = []
    for gi in range(args.groups):
        gid = f"g{gi + 1}"
        members = picked[gi * args.x:(gi + 1) * args.x]
        groups.append({"id": gid, "images": members})
        for iid in members:
            cls = class_of(iid, names, lbl_dir)
            for arm in ARMS:
                tasks.append({
                    "image_id": iid,
                    "arm": arm,
                    "group": gid,
                    "class_name": cls,
                })

    ordered = build_order(tasks, args.layout, args.x, args.groups, rnd)

    manifest = {
        "name": args.name or (
            f"同图三臂 · {args.x}图×{args.groups}组"
            + (f" · 第{args.person}人" if args.persons > 1 else "")
        ),
        "design": {
            "kind": "same-image-3arms",
            "x": args.x,
            "groups": args.groups,
            "arms": list(ARMS),
            "split": args.split,
            "seed": args.seed,
            "layout": args.layout,
            "persons": args.persons,
            "person": args.person,
            "total_tasks": len(ordered),
            "arm_order_rule": "by-arm 时每组的臂顺序按 (组号 + k) mod 3 轮转",
            "note": "同一张图在三个臂各标一遍；不同 person 的图互不重叠（避免同臂同图覆盖）",
        },
        "groups": groups,
        "tasks": ordered,
    }

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 自检
    per_arm = Counter(t["arm"] for t in ordered)
    per_group = Counter(t["group"] for t in ordered)
    seq = []
    for g in sorted(per_group, key=lambda s: int(s[1:])):
        block = [t["arm"] for t in ordered if t["group"] == g]
        # 压缩成"臂块"序列，便于肉眼检查轮转
        blocks = [a for i, a in enumerate(block) if i == 0 or block[i - 1] != a]
        seq.append(f"{g}:" + "→".join(b[0].upper() for b in blocks))
    cls_img: dict[str, str] = {}
    for t in ordered:
        cls_img.setdefault(t["image_id"], t["class_name"] or "（无标签）")

    print(f"  清单 -> {out}")
    print(f"  {args.x} 图 × 3 臂 × {args.groups} 组 = {len(ordered)} 个任务"
          f"（{len(cls_img)} 张图）")
    print(f"  标注员: 第 {args.person}/{args.persons} 人"
          + (f"（本套图与他人不重叠，总抽样 {need} 张）" if args.persons > 1 else ""))
    print(f"  每臂任务数: {dict(sorted(per_arm.items()))}")
    print(f"  每组任务数: {dict(sorted(per_group.items(), key=lambda kv: int(kv[0][1:])))}")
    print(f"  组内臂块顺序: {' '.join(seq)}")
    print(f"  布局: {args.layout}   seed: {args.seed}")
    print(f"  覆盖类别: {len(set(cls_img.values()))} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
