"""跨路由共用的小工具。"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

_UNSAFE = re.compile(r"[^A-Za-z0-9._\-\u4e00-\u9fff]")


def safe_name(s: str, fallback: str = "unnamed") -> str:
    """把任意字符串变成可安全作文件名 / 目录名的片段。

    允许中文（标注员可能用中文命名臂），其余非法字符换成下划线；
    顺带掐掉前导的点，避免生成 . / .. 这类特殊条目。
    """
    out = _UNSAFE.sub("_", str(s)).lstrip(".")
    return out[:120] or fallback


def now_stamp() -> str:
    """本地时间，ISO 8601 到秒。用于 saved_at / pred_ts / ts。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """先写 .tmp 再 os.replace —— 半截文件不会被读到（m5/m6 会读这些文件）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding=encoding, newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def resolve_arm(arm: str | None) -> str:
    """解析并校验实验臂（annotations 与 logs 共用）。

    为什么要白名单：`out/annotations/<arm>/` 与 `out/logs/ops_<arm>.csv` 都按臂分，
    臂名写错一个字母（带空格、大写）就会静默多出一份数据，三臂对照的分母就错了，
    而且事后极难发现。宁可当场 422。
    """
    from fastapi import HTTPException

    from . import config

    name = safe_name(arm or config.DEFAULT_ARM, fallback=config.DEFAULT_ARM)
    if name not in config.VALID_ARMS:
        raise HTTPException(
            422,
            f"未知实验臂 {name!r}，可选 {list(config.VALID_ARMS)}。"
            f"臂名决定产物落到哪个目录/文件，写错会毁掉对照实验",
        )
    return name
