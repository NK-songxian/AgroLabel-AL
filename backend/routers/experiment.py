"""实验模式：把三臂对照实验的图清单（manifest）暴露给前端。

为什么需要它
------------
三臂实验只标清单里的 68 张 val 图，而且**每张图绑定一个臂**。
没有这个接口，标注员得在 7773 张的队列里自己找图、自己切臂 —— 几乎必错，
而臂标错的数据是废的（产物落到 out/annotations/<arm>/，ops 也按臂分文件）。

接口**只读**：读清单、数一下哪些图已经标过，不写任何东西。

清单位置
--------
默认 `out/experiment/manifest.json`，可用环境变量 `AGROLABEL_EXPERIMENT` 指向别处。
格式兼容 exp60 的产物::

    {"experiment": "...", "images": [{"image_id": "...", "arm_backend": "manual"}, ...]}

`arm_backend` 与 `arm` 都认（前者是 exp60 的字段名）；未知臂名、重复 id
都会记进 `problems` 并跳过，不会让整个接口失败。
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
MAX_IMAGES = 2000


class ExpImage(BaseModel):
    image_id: str
    arm: str
    class_name: str | None = None
    done: bool = False


class ExperimentOut(BaseModel):
    loaded: bool
    source: str | None = None
    name: str | None = None
    note: str = ""
    n_images: int = 0
    arms: dict[str, int] = Field(default_factory=dict)
    done: dict[str, int] = Field(default_factory=dict)
    done_images: list[str] = Field(default_factory=list)
    images: list[ExpImage] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)


def manifest_path() -> Path:
    """清单路径。环境变量优先，方便实验时指到别处而不改代码。"""
    raw = os.environ.get(MANIFEST_ENV, "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_MANIFEST


def _display(path: Path) -> str:
    """能在日志/界面上显示的路径：项目内的相对路径，项目外给绝对路径。"""
    try:
        return path.resolve().relative_to(config.PROJECT).as_posix()
    except ValueError:
        return str(path)


def _annotated(arm: str, image_id: str) -> bool:
    """这张图在该臂下有没有已落盘的标注（sidecar 存在即算，内容不校验）。"""
    return (config.ANNOTATION_DIR / arm / f"{safe_name(image_id)}.json").is_file()


def _not_loaded(note: str, source: str | None = None) -> ExperimentOut:
    return ExperimentOut(loaded=False, source=source, note=note)


def read_manifest() -> ExperimentOut:
    """解析清单。返回体里 `loaded=false` 表示"没有可用清单"，不是错误。"""
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

    raw_images = data.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        return _not_loaded("实验清单里没有 images 列表", shown)

    images: list[ExpImage] = []
    problems: list[str] = []
    seen: set[str] = set()

    for i, item in enumerate(raw_images):
        if not isinstance(item, dict):
            problems.append(f"images[{i}] 不是对象，跳过")
            continue
        iid = str(item.get("image_id") or "").strip()
        arm = str(item.get("arm_backend") or item.get("arm") or "").strip()
        if not iid:
            problems.append(f"images[{i}] 缺 image_id，跳过")
            continue
        if arm not in config.VALID_ARMS:
            problems.append(f"{iid}: 未知臂 {arm!r}，应为 {list(config.VALID_ARMS)}，跳过")
            continue
        if iid in seen:
            problems.append(f"{iid}: 清单里重复出现，只取第一条")
            continue
        seen.add(iid)
        cls = item.get("gt_class_name") or item.get("class_name")
        images.append(
            ExpImage(
                image_id=iid,
                arm=arm,
                class_name=str(cls) if cls else None,
                done=_annotated(arm, iid),
            )
        )
        if len(images) >= MAX_IMAGES:
            problems.append(f"超过 {MAX_IMAGES} 张，其余忽略")
            break

    if not images:
        return _not_loaded("清单里没有一张可用的图（见 problems）", shown)

    arms = {a: 0 for a in config.VALID_ARMS}
    done = {a: 0 for a in config.VALID_ARMS}
    for it in images:
        arms[it.arm] += 1
        if it.done:
            done[it.arm] += 1
    done_images = [it.image_id for it in images if it.done]

    name = data.get("name") or data.get("experiment")
    return ExperimentOut(
        loaded=True,
        source=shown,
        name=str(name) if name else path.stem,
        note=f"清单 {len(images)} 张，已标 {len(done_images)} 张",
        n_images=len(images),
        arms=arms,
        done=done,
        done_images=done_images,
        images=images,
        problems=problems,
    )


@router.get("/experiment", response_model=ExperimentOut)
def get_experiment() -> ExperimentOut:
    """当前实验清单 + 每臂进度。前端据此把队列换成这 68 张、并按图自动切臂。"""
    return read_manifest()
