"""POST /api/logs/ops —— 上报操作日志；GET /api/logs/export —— 导出（P3 占位）

契约：docs/INTERFACE_ADDENDUM.md §2.5 / §2.8

为什么表头和词表必须逐字对齐：src/m5_logstats.py 用 csv.DictReader 按列名读，
且把"改类别 / 删实例 / 标为无目标"硬编码成计入修改次数的集合。表头错一个字母、
op 写成一个新词，那 25 分里的量化项就直接失准。

    ts, image, op, instance, detail, n_ops_on_image, api_ms

编码固定 utf-8-sig —— m4_app.py 写、m5 用 utf-8-sig 读，带中文的 detail 才不会乱码。
"""

from __future__ import annotations

import csv
import json
import threading
import time

from fastapi import APIRouter, HTTPException, Query, Response

from .. import config
from ..schemas import OpsRequest, OpsResponse
from ..util import resolve_arm

router = APIRouter(prefix="/api", tags=["logs"])

HEADER = ["ts", "image", "op", "instance", "detail", "n_ops_on_image", "api_ms"]

# 与 src/m5_logstats.py + src/m4_app.py 的词表一致；不在这张表里的 op 一律打回
COUNTED_OPS = {"改类别", "删实例", "标为无目标"}
NEUTRAL_OPS = {"整张确认", "撤销", "提交"}
EXTENDED_OPS = {"新画", "拖边界"}
ALLOWED_OPS = COUNTED_OPS | NEUTRAL_OPS | EXTENDED_OPS

# 单进程内串行化追加，避免并发请求把行写串
_append_lock = threading.Lock()


@router.post("/logs/ops", response_model=OpsResponse)
def append_ops(
    body: OpsRequest,
    arm: str | None = Query(
        default=None,
        description="实验臂；同时写进 out/logs/ops_<arm>.csv，供按臂统计",
    ),
):
    bad = sorted({e.op for e in body.events if e.op not in ALLOWED_OPS})
    if bad:
        raise HTTPException(
            422,
            f"op 不在词表内: {bad}。允许: {sorted(ALLOWED_OPS)}；"
            f"词表由 src/m5_logstats.py 硬编码，新增必须先改那边",
        )

    arm_name = resolve_arm(arm)
    path = config.OPS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    # 同一个臂的日志再写一份分臂文件，表头完全一样。
    # 为什么不在 ops.csv 里加一列 arm：那份表头要和 src/m4_app.py:131 逐字对齐
    # （契约要求可直接追加），改表头会让两边的行混不成一张表。
    arm_path = path.with_name(f"ops_{arm_name}.csv")

    with _append_lock:
        for target in (path, arm_path):
            write_header = not target.exists() or target.stat().st_size == 0
            # newline="" 是 csv 模块的硬要求，否则 Windows 上会多出空行
            with open(target, "a", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(HEADER)
                for e in body.events:
                    w.writerow([
                        e.ts, e.image, e.op, e.instance,
                        e.detail, e.n_ops_on_image, e.api_ms,
                ])
        rows_total = _count_rows(path)

    return OpsResponse(
        ok=True,
        appended=len(body.events),
        rows_total=rows_total,
        path=str(path.relative_to(config.PROJECT)).replace("\\", "/"),
    )


def _count_rows(path) -> int:
    """不含表头的数据行数。给前端一个进度感，也方便排查重复上报。"""
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


@router.get("/logs/export")
def export_ops(
    format: str = "csv",  # noqa: A002 —— 契约里就叫 format
    arm: str | None = Query(
        default=None,
        description="实验臂；给了就导出 out/logs/ops_<arm>.csv，不给导合并的 ops.csv",
    ),
):
    """导出操作日志（ADDENDUM §2.8）。

    `format=csv` 返回原文件字节流 —— **不做重新编码**，保留 UTF-8 BOM，
    这样 Excel 双击打开中文表头/中文 detail 不会乱码（m4/m5 也是这个约定）。

    `format=json` 解析成对象数组，供脚本直接消费（pandas.read_json 之类）。

    三个设计取舍：
      - 带 `arm` 就导分臂文件。三臂实验要按臂出数据，合并的 ops.csv 分不开。
      - 文件不存在时返回 **200 + 只有表头的空 CSV**，而不是 404：
        导出是"下载文件"的语义，给一个能打开的合法 CSV 比报错更有用。
      - `Content-Disposition` 带时间戳，连续导两次不会互相覆盖。
    """
    fmt = (format or "csv").lower()
    if fmt not in ("csv", "json"):
        raise HTTPException(400, f"format={fmt} 不支持，可选 csv / json")

    src = config.OPS_CSV
    if arm:
        src = src.with_name(f"ops_{resolve_arm(arm)}.csv")

    rows: list[dict] = []
    if src.is_file():
        with open(src, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = src.stem                             # ops 或 ops_<arm>
    filename = f"{stem}_{stamp}.{fmt}"

    if fmt == "json":
        body = json.dumps(
            {"source": src.name, "rows": len(rows), "events": rows},
            ensure_ascii=False, indent=2,
        )
        return Response(
            content=body,
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    if src.is_file():
        # 原样回传，保留 BOM —— 重新编码会破坏 Excel 的兼容性
        payload = src.read_bytes()
    else:
        # 给一个只有表头的合法 CSV，前端/Excel 都能正常打开
        payload = ("\ufeff" + ",".join(HEADER) + "\n").encode("utf-8")

    return Response(
        content=payload,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Row-Count": str(len(rows)),
        },
    )
