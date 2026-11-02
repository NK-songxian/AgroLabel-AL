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
import threading

from fastapi import APIRouter, HTTPException

from .. import config
from ..schemas import OpsRequest, OpsResponse

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
def append_ops(body: OpsRequest):
    bad = sorted({e.op for e in body.events if e.op not in ALLOWED_OPS})
    if bad:
        raise HTTPException(
            422,
            f"op 不在词表内: {bad}。允许: {sorted(ALLOWED_OPS)}；"
            f"词表由 src/m5_logstats.py 硬编码，新增必须先改那边",
        )

    path = config.OPS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)

    with _append_lock:
        write_header = not path.exists() or path.stat().st_size == 0
        # newline="" 是 csv 模块的硬要求，否则 Windows 上会多出空行
        with open(path, "a", encoding="utf-8-sig", newline="") as f:
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
def export_ops(format: str = "csv"):  # noqa: A002 —— 契约里就叫 format
    """导出 ops.csv。

    团队决议（ADDENDUM §7 第 4 条）：**留位置，P3 再做**。
    这里注册路由并**一律返回 501** + 明确说明，不做成 404 ——
    404 会被前端当成"地址打错"，501 才是"接口存在但没实现"。
    """
    if format != "csv":
        raise HTTPException(400, f"format={format} 不支持，目前只有 csv")

    where = "尚未产生任何日志" if not config.OPS_CSV.is_file() else \
        f"日志已在 {str(config.OPS_CSV.relative_to(config.PROJECT)).replace(chr(92), '/')}" \
        f"（{config.OPS_CSV.stat().st_size} 字节），可直接取用"

    raise HTTPException(
        501,
        f"未实现：/api/logs/export 排在 P3。{where}。"
        f"当前导出走前端右栏的『下载 ops.csv』，或直接读该文件",
    )
