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

import json
import os
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .. import config
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


@router.get("/experiment", response_model=ExperimentOut)
def get_experiment() -> ExperimentOut:
    """当前实验任务清单 + 每臂/每臂组进度。"""
    return read_manifest()
