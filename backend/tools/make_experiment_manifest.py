"""生成「同图三臂」实验任务清单（命令行入口）。

真正的构造逻辑在 `backend/experiment_manifest.py`，后端 `/api/experiment/reroll`
也用同一份，避免两处抽样规则漂移。

用法
----
    # 单人 8 图 × 4 组
    python backend/tools/make_experiment_manifest.py --data .../data.yaml --x 8 --groups 4

    # 两个人，各 8×4，互不重叠
    ... --persons 2 --person 1 --out out/experiment/p1.json
    ... --persons 2 --person 2 --out out/experiment/p2.json

    # 抗疲劳更强（严格交错）
    ... --layout interleave
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))          # 让 `import backend.*` 在脚本模式下可用

from backend.experiment_manifest import (  # noqa: E402
    build_manifest,
    dataset_root,
    summarize,
    write_manifest,
)


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
    ap.add_argument("--layout", choices=("by-arm", "interleave", "by-group"),
                    default="by-arm",
                    help="by-arm=一组内按臂分块（默认）；interleave=全随机交错")
    ap.add_argument("--out", default=str(PROJECT / "out" / "experiment" / "manifest.json"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--exclude", nargs="*", default=[], help="排除这些 image_id")
    args = ap.parse_args()

    data_yaml = Path(args.data).expanduser()
    if not data_yaml.is_file():
        print(f"找不到 data.yaml: {data_yaml}", file=sys.stderr)
        return 2
    root = dataset_root(data_yaml, data_yaml.parent)

    try:
        manifest = build_manifest(
            root=root, data_yaml=data_yaml, split=args.split,
            x=args.x, groups=args.groups, persons=args.persons, person=args.person,
            seed=args.seed, layout=args.layout, exclude=set(args.exclude),
            name=args.name,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    out = write_manifest(manifest, Path(args.out).expanduser())
    s = summarize(manifest)
    d = manifest["design"]
    print(f"  清单 -> {out}")
    print(f"  {d['x']} 图 × 3 臂 × {d['groups']} 组 = {s['n_tasks']} 个任务"
          f"（{s['n_images']} 张图）")
    print(f"  标注员: 第 {d['person']}/{d['persons']} 人"
          + ("（本套图与他人不重叠）" if d["persons"] > 1 else ""))
    print(f"  每臂任务数: {s['per_arm']}")
    print(f"  每组任务数: {s['per_group']}")
    print(f"  组内臂块顺序: {' '.join(s['arm_blocks'])}")
    print(f"  布局: {d['layout']}   seed: {d['seed']}")
    print(f"  覆盖类别: {s['n_classes']} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
