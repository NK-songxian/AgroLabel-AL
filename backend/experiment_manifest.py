"""实验任务清单的构造逻辑（后端接口与命令行工具共用）。

放在 backend 顶层（而不是 backend/tools 里），这样 `routers/experiment.py`
可以直接 import，不用给 tools 建包，也不用 subprocess 调脚本。

设计要点见 docs/实验安排_同图三臂.md：
- 同图三臂：一张图在 3 个臂各标一遍
- 臂优先：组内按臂分块，臂顺序按组号轮转
- 多人不重叠：--persons N --person I 从同一抽样里切片，避免同臂同图互相覆盖
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

ARMS = ("assisted", "manual", "noprompt")


def load_names(data_yaml: Path | None) -> list[str]:
    """读 data.yaml 的 names（只用于清单里显示类别名）。"""
    if data_yaml is None or not Path(data_yaml).is_file():
        return []
    try:
        import yaml

        conf = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    names = conf.get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(x) for x in names]


def dataset_root(data_yaml: Path | None, fallback: Path) -> Path:
    """data.yaml 里 `path:` 优先；否则用 yaml 所在目录。"""
    if data_yaml is None:
        return fallback
    try:
        import yaml

        conf = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
        declared = Path(str(conf["path"])) if conf.get("path") else None
        if declared and declared.is_dir():
            return declared
    except Exception:
        pass
    return Path(data_yaml).parent


def class_of(image_id: str, names: list[str], labels_dir: Path | None) -> str | None:
    if labels_dir is None or not names:
        return None
    lbl = labels_dir / f"{image_id}.txt"
    if not lbl.is_file():
        return None
    try:
        cid = int(lbl.read_text(encoding="utf-8").splitlines()[0].split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return names[cid] if 0 <= cid < len(names) else None


def build_order(tasks: list[dict], layout: str, groups: int,
                rnd: random.Random) -> list[dict]:
    """决定执行顺序并写 order（1-based）。默认 by-arm：组内按臂分块。"""
    if layout == "interleave":
        out = list(tasks)
        rnd.shuffle(out)
    elif layout == "by-group":
        out = []
        for gi in range(groups):
            block = [t for t in tasks if t["group"] == f"g{gi + 1}"]
            rnd.shuffle(block)
            out.extend(block)
    else:  # by-arm
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


def build_manifest(
    *,
    root: Path,
    data_yaml: Path | None = None,
    split: str = "val",
    x: int = 8,
    groups: int = 4,
    persons: int = 1,
    person: int = 1,
    seed: int = 20261124,
    layout: str = "by-arm",
    exclude: set[str] | None = None,
    name: str | None = None,
) -> dict:
    """抽图 + 分组 + 建任务，返回清单 dict（不落盘）。

    失败抛 ValueError（消息可直接给前端）。
    """
    if x < 1 or groups < 1 or persons < 1:
        raise ValueError("x / groups / persons 必须 >= 1")
    if not 1 <= person <= persons:
        raise ValueError(f"person 必须在 1..{persons} 之间")

    img_dir = root / "images" / split
    if not img_dir.is_dir():
        raise ValueError(f"找不到图片目录: {img_dir}")
    names = load_names(data_yaml)
    lbl_dir = root / "labels" / split

    pool = sorted(p.stem for p in img_dir.glob("*.jpg"))
    skip = exclude or set()
    pool = [i for i in pool if i not in skip]
    per_person = x * groups
    need = per_person * persons
    if len(pool) < need:
        raise ValueError(
            f"可用图 {len(pool)} 张，不够 {need} 张"
            f"（x={x} × groups={groups} × persons={persons}）"
        )

    rnd = random.Random(seed)
    sampled = rnd.sample(pool, need)
    picked = sampled[(person - 1) * per_person: person * per_person]

    grp_meta: list[dict] = []
    tasks: list[dict] = []
    for gi in range(groups):
        gid = f"g{gi + 1}"
        members = picked[gi * x:(gi + 1) * x]
        grp_meta.append({"id": gid, "images": members})
        for iid in members:
            cls = class_of(iid, names, lbl_dir)
            for arm in ARMS:
                tasks.append({
                    "image_id": iid, "arm": arm, "group": gid, "class_name": cls,
                })

    ordered = build_order(tasks, layout, groups, rnd)
    return {
        "name": name or (
            f"同图三臂 · {x}图×{groups}组"
            + (f" · 第{person}人" if persons > 1 else "")
        ),
        "design": {
            "kind": "same-image-3arms",
            "x": x, "groups": groups, "arms": list(ARMS), "split": split,
            "seed": seed, "layout": layout, "persons": persons, "person": person,
            "total_tasks": len(ordered),
            "arm_order_rule": "by-arm 时每组的臂顺序按 (组号 + k) mod 3 轮转",
            "note": "同一张图在三个臂各标一遍；不同 person 的图互不重叠（避免同臂同图覆盖）",
        },
        "groups": grp_meta,
        "tasks": ordered,
    }


def write_manifest(manifest: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def summarize(manifest: dict) -> dict:
    """给命令行/接口共用的一行统计。"""
    tasks = manifest["tasks"]
    per_arm = Counter(t["arm"] for t in tasks)
    per_group = Counter(t["group"] for t in tasks)
    seq = []
    for g in sorted(per_group, key=lambda s: int(s[1:])):
        block = [t["arm"] for t in tasks if t["group"] == g]
        blocks = [a for i, a in enumerate(block) if i == 0 or block[i - 1] != a]
        seq.append(f"{g}:" + "→".join(b[0].upper() for b in blocks))
    return {
        "n_tasks": len(tasks),
        "n_images": len({t["image_id"] for t in tasks}),
        "per_arm": dict(sorted(per_arm.items())),
        "per_group": dict(sorted(per_group.items(), key=lambda kv: int(kv[0][1:]))),
        "arm_blocks": seq,
        "n_classes": len({t["class_name"] for t in tasks if t["class_name"]}),
    }
