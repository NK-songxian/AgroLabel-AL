"""实验模式：把三臂对照实验的**任务清单**暴露给前端。

两种清单形态（按顺序尝试，向后兼容）
------------------------------------
1. `tasks` —— 每个任务 = (图, 臂)。**同图三臂**设计靠这个：同一张图在三个臂各标一遍，
   图难度被完全消掉，每张图给出一个配对差。
2. `images` —— 每个图只属于一个臂（旧形态，exp60 那种）。

```json
{
  "name": "同图三臂 · 8图×4组",
  "tasks": [
    {"image_id": "grape_black_rot_141", "arm": "manual",   "group": "g1", "order": 1},
    {"image_id": "grape_black_rot_141", "arm": "assisted", "group": "g1", "order": 2}
  ]
}
```

`order` 是全局执行顺序（前端按它走路）；`group` 只用于事后按组汇总。
字段都可省略，但同图三臂必须写 `arm`，否则同一张图的三个任务分不开。

为什么需要它
------------
没有这个接口，标注员得在 7773 张的队列里自己找图、自己切臂 —— 几乎必错，
而臂标错的数据是废的（产物落到 out/annotations/<arm>/，ops 也按臂分文件）。

接口**只读**：读清单、数一下哪些任务已经标过，不写任何东西。
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import config
from ..experiment_manifest import build_manifest, write_manifest
from ..util import safe_name

router = APIRouter(prefix="/api", tags=["experiment"])

MANIFEST_ENV = "AGROLABEL_EXPERIMENT"
DEFAULT_MANIFEST = config.OUT_DIR / "experiment" / "manifest.json"
MAX_TASKS = 5000


class ExpTask(BaseModel):
    image_id: str
    arm: str
    group: str | None = None
    order: int | None = None
    class_name: str | None = None
    done: bool = False


class ExpImage(BaseModel):
    image_id: str
    class_name: str | None = None


class ExperimentOut(BaseModel):
    loaded: bool
    mode: str = "none"  # tasks | images | none
    source: str | None = None
    name: str | None = None
    note: str = ""
    n_tasks: int = 0
    n_images: int = 0
    arms: dict[str, int] = Field(default_factory=dict)
    done: dict[str, int] = Field(default_factory=dict)
    done_tasks: list[str] = Field(default_factory=list)
    groups: dict[str, dict[str, int]] = Field(default_factory=dict)
    images: list[ExpImage] = Field(default_factory=list)
    tasks: list[ExpTask] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)


def manifest_path() -> Path:
    """清单路径。环境变量优先，方便实验时指到别处而不改代码。"""
    raw = os.environ.get(MANIFEST_ENV, "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_MANIFEST


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(config.PROJECT).as_posix()
    except ValueError:
        return str(path)


def _annotated(arm: str, image_id: str) -> bool:
    """该 (图, 臂) 有没有已落盘标注。sidecar 存在即算，不校验内容。"""
    return (config.ANNOTATION_DIR / arm / f"{safe_name(image_id)}.json").is_file()


def task_key(image_id: str, arm: str) -> str:
    """前端也用这个拼 key：一张图会有多个任务，光靠 image_id 分不开。"""
    return f"{image_id}|{arm}"


def _not_loaded(note: str, source: str | None = None) -> ExperimentOut:
    return ExperimentOut(loaded=False, mode="none", source=source, note=note)


def _parse_items(raw: list, problems: list[str]) -> list[ExpTask]:
    """把 tasks / images 两种条目统一解析成 ExpTask。"""
    out: list[ExpTask] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            problems.append(f"[{i}] 不是对象，跳过")
            continue
        iid = str(item.get("image_id") or "").strip()
        arm = str(item.get("arm_backend") or item.get("arm") or "").strip()
        if not iid:
            problems.append(f"[{i}] 缺 image_id，跳过")
            continue
        if arm not in config.VALID_ARMS:
            problems.append(f"{iid}: 未知臂 {arm!r}，应为 {list(config.VALID_ARMS)}，跳过")
            continue
        key = task_key(iid, arm)
        if key in seen:
            # 同图同臂重复 = 同一任务重复，直接去重（重复跑一遍会覆盖前一遍）
            problems.append(f"{key}: 任务重复，只取第一条")
            continue
        seen.add(key)
        cls = item.get("gt_class_name") or item.get("class_name")
        grp = item.get("group")
        order = item.get("order")
        out.append(
            ExpTask(
                image_id=iid,
                arm=arm,
                group=str(grp) if grp not in (None, "") else None,
                order=int(order) if isinstance(order, (int, float)) else None,
                class_name=str(cls) if cls else None,
                done=_annotated(arm, iid),
            )
        )
        if len(out) >= MAX_TASKS:
            problems.append(f"超过 {MAX_TASKS} 个任务，其余忽略")
            break
    return out


def read_manifest() -> ExperimentOut:
    """解析清单。`loaded=false` 表示"没有可用清单"，不是错误。"""
    path = manifest_path()
    shown = _display(path)
    if not path.is_file():
        return _not_loaded(
            f"没有实验清单：{shown}（可用环境变量 {MANIFEST_ENV} 指定路径）",
            shown,
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _not_loaded(f"实验清单解析失败：{exc}", shown)

    if not isinstance(data, dict):
        return _not_loaded("实验清单顶层必须是对象", shown)

    problems: list[str] = []
    raw_tasks = data.get("tasks")
    if isinstance(raw_tasks, list) and raw_tasks:
        mode = "tasks"
        eff = _parse_items(raw_tasks, problems)
    else:
        raw_images = data.get("images")
        if not isinstance(raw_images, list) or not raw_images:
            return _not_loaded("实验清单里没有 tasks 或 images 列表", shown)
        mode = "images"
        eff = _parse_items(raw_images, problems)

    if not eff:
        return _not_loaded("清单里没有一条可用的任务（见 problems）", shown)

    # 有 order 就按它排；没有就保持文件顺序
    if any(t.order is not None for t in eff):
        eff.sort(key=lambda t: (t.order is None, t.order or 0))

    arms = {a: 0 for a in config.VALID_ARMS}
    done = {a: 0 for a in config.VALID_ARMS}
    groups: dict[str, dict[str, int]] = {}
    for t in eff:
        arms[t.arm] += 1
        if t.done:
            done[t.arm] += 1
        g = groups.setdefault(t.group or "未分组", {"n": 0, "done": 0})
        g["n"] += 1
        if t.done:
            g["done"] += 1

    uniq: dict[str, ExpImage] = {}
    for t in eff:
        uniq.setdefault(t.image_id, ExpImage(image_id=t.image_id, class_name=t.class_name))

    done_tasks = [task_key(t.image_id, t.arm) for t in eff if t.done]
    name = data.get("name") or data.get("experiment")
    unit = "任务" if mode == "tasks" else "图"
    return ExperimentOut(
        loaded=True,
        mode=mode,
        source=shown,
        name=str(name) if name else path.stem,
        note=f"清单 {len(eff)} 个{unit}（{len(uniq)} 张图），已完成 {len(done_tasks)} 个",
        n_tasks=len(eff),
        n_images=len(uniq),
        arms=arms,
        done=done,
        done_tasks=done_tasks,
        groups=groups,
        images=list(uniq.values()),
        tasks=eff,
        problems=problems,
    )


class RerollIn(BaseModel):
    """重新随机。

    `confirm=False` 只做预演（不改任何文件），用来告诉前端"会删多少"。
    """

    confirm: bool = False
    seed: int | None = None


class RerollOut(BaseModel):
    dry_run: bool
    n_tasks: int
    n_done: int
    n_sidecars: int
    n_ops_rows: int
    n_pred_cache: int
    archived_to: str | None = None
    seed: int | None = None
    name: str | None = None
    note: str = ""


def _ops_files() -> list[Path]:
    """操作日志按臂分文件：out/logs/ops_<arm>.csv（表头 ts,image,op,...）。"""
    return [config.LOG_DIR / f"ops_{arm}.csv" for arm in config.VALID_ARMS]


def _count_ops_rows(ids: set[str]) -> int:
    n = 0
    for f in _ops_files():
        if not f.is_file():
            continue
        try:
            with f.open(encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    if (row.get("image") or "") in ids:
                        n += 1
        except OSError:
            continue
    return n


def _filter_ops(ids: set[str], archive: Path) -> int:
    """把属于本轮图的操作行摘掉（原文件先备份到 archive）。返回摘掉的行数。"""
    removed = 0
    for f in _ops_files():
        if not f.is_file():
            continue
        try:
            with f.open(encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                fields = reader.fieldnames or []
                keep, drop = [], []
                for row in reader:
                    (drop if (row.get("image") or "") in ids else keep).append(row)
        except OSError:
            continue
        if not drop:
            continue
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, archive / f.name)          # 备份原始 CSV
        with f.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(keep)
        removed += len(drop)
    return removed


def _sidecars(targets: set[tuple[str, str]]) -> list[Path]:
    out = []
    for iid, arm in targets:
        for suf in (".json", ".txt"):
            p = config.ANNOTATION_DIR / arm / f"{safe_name(iid)}{suf}"
            if p.is_file():
                out.append(p)
    return out


def _pred_caches(ids: set[str]) -> list[Path]:
    out = []
    for iid in ids:
        p = config.PRED_CACHE_DIR / f"{safe_name(iid)}.json"
        if p.is_file():
            out.append(p)
    return out


@router.post("/experiment/reroll", response_model=RerollOut)
def reroll(body: RerollIn) -> RerollOut:
    """重新随机抽图，并清掉本轮未完成的标注。

    为什么要"清"：换了一批图之后，旧图上的部分标注既不属于新清单，
    又会污染 `out/annotations/<arm>/` 和 ops 统计。所以：

    1. 先把要清的东西**备份**到 `out/experiment/archive/<时间戳>/`（可回退）
    2. 再移走 sidecar（`.json` + `.txt`）、摘掉 ops 里这些图的行、清 pred_cache
    3. 最后按同样的 design（x/groups/layout…）换 seed 重新抽一份清单

    `confirm=false` 时只统计不落盘 —— 前端先让用户看清"会删多少"再确认。
    """
    path = manifest_path()
    cur = read_manifest()
    if not cur.loaded:
        raise HTTPException(409, cur.note or "没有实验清单，无法重新随机")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"清单读取失败：{exc}")
    design = raw.get("design") or {}

    targets = {(t.image_id, t.arm) for t in cur.tasks}
    ids = {iid for iid, _ in targets}
    sidecars = _sidecars(targets)
    preds = _pred_caches(ids)
    ops_rows = _count_ops_rows(ids)
    n_done = sum(1 for t in cur.tasks if t.done)

    if not body.confirm:
        return RerollOut(
            dry_run=True, n_tasks=len(cur.tasks), n_done=n_done,
            n_sidecars=len(sidecars), n_ops_rows=ops_rows,
            n_pred_cache=len(preds),
            note="预演：还没有动任何文件",
        )

    # ---- 1) 备份 + 清理
    archive = (config.OUT_DIR / "experiment" / "archive"
               / datetime.now().strftime("%Y%m%d-%H%M%S"))
    for p in sidecars:
        dest = archive / "annotations" / p.parent.name / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dest))
    for p in preds:
        dest = archive / "pred_cache" / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dest))
    _filter_ops(ids, archive)

    # ---- 2) 换 seed 重新抽图
    try:
        base_seed = int(design.get("seed") or 20261124)
    except (TypeError, ValueError):
        base_seed = 20261124
    new_seed = body.seed if body.seed is not None else base_seed + 1

    root = config.DATA.root
    data_yaml = root / "data.yaml"
    try:
        manifest = build_manifest(
            root=root,
            data_yaml=data_yaml if data_yaml.is_file() else None,
            split=str(design.get("split") or "val"),
            x=int(design.get("x") or 8),
            groups=int(design.get("groups") or 4),
            persons=int(design.get("persons") or 1),
            person=int(design.get("person") or 1),
            seed=new_seed,
            layout=str(design.get("layout") or "by-arm"),
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"重新抽图失败：{exc}") from exc

    write_manifest(manifest, path)

    rel = archive
    try:
        rel = archive.relative_to(config.PROJECT)
    except ValueError:
        pass
    return RerollOut(
        dry_run=False, n_tasks=len(cur.tasks), n_done=n_done,
        n_sidecars=len(sidecars), n_ops_rows=ops_rows, n_pred_cache=len(preds),
        archived_to=str(rel).replace("\\", "/"),
        seed=new_seed, name=manifest.get("name"),
        note=f"已备份到 {rel}，并按 seed={new_seed} 重新抽图",
    )


@router.get("/experiment", response_model=ExperimentOut)
def get_experiment() -> ExperimentOut:
    """当前实验任务清单 + 每臂/每臂组进度。"""
    return read_manifest()
