"""POST /api/sample —— 主动学习选样

契约：docs/INTERFACE.md §2、ADDENDUM §3.2

算法**不在这里**：直接调模型侧 `active_learning/select.py::select()`，
这样离线消融实验和在线接口用的是同一套打分口径，报告里的数字才对得上。
本路由只负责三件它不适合管的事：

  1. image_id  <-> 磁盘路径  的映射
  2. 默认 pool：整个 catalog 并剔除已标注过的图（否则反复点会一直推荐同一批）
  3. reasons 的结构化（契约 §3.2 要求前端能展示"为什么选它"）
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request

from .. import config
from ..schemas import SampleReason, SampleRequest, SampleResponse
from ..util import safe_name

router = APIRouter(prefix="/api", tags=["sample"])

MAX_POOL = 500          # 单次最多打分这么多张，防止一个请求跑上几分钟


@router.post("/sample", response_model=SampleResponse)
def sample(body: SampleRequest, request: Request):
    catalog = request.app.state.catalog

    # ---- 1. 确定 pool
    if body.pool:
        unknown = [i for i in body.pool if catalog.get(i) is None]
        if unknown:
            raise HTTPException(
                404,
                f"pool 里有 {len(unknown)} 个未知 image_id，例如 {unknown[:5]}；"
                f"务必把提交时的 image_id 原样传回",
            )
        ids = list(dict.fromkeys(body.pool))       # 去重但保序
    else:
        ids = [r.image_id for r in catalog.query(limit=10_000)[0]]
        if body.exclude_annotated:
            done = _annotated_ids()
            ids = [i for i in ids if i not in done]

    if not ids:
        raise HTTPException(
            400,
            "可选池为空：所有图都标过了，或 pool 传成了空数组。"
            "要重新过一遍请传 exclude_annotated=false",
        )
    if len(ids) > MAX_POOL:
        raise HTTPException(
            400, f"pool 有 {len(ids)} 张，超过单次上限 {MAX_POOL}；请分批调用"
        )

    paths = [catalog.get(i).path for i in ids]
    # 注意：select() 内部会把 pool 转成 str 再返回，所以这里键也用 str，
    # 否则 Path 键和 str 查不到一起，每次都要走 O(n) 的兜底扫描。
    id_of = {str(p): i for p, i in zip(paths, ids)}

    # ---- 2. 交给模型侧选样
    t0 = time.perf_counter()
    try:
        from active_learning.select import select as al_select
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(
            501, f"模型侧 active_learning/select.py 导入失败：{type(e).__name__}: {e}"
        )
    try:
        picked = al_select(
            [str(p) for p in paths],
            body.k,
            body.strategy,
            weights=str(config.WEIGHTS),
            imgsz=config.IMGSZ,
            device=config.resolve_device(),
            seed=body.seed,
        )
    except FileNotFoundError as e:
        raise HTTPException(500, f"选样缺少权重：{e}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    latency = (time.perf_counter() - t0) * 1000

    # ---- 3. 路径映射回 image_id，并按顺序给 rank
    selected: list[str] = []
    reasons: dict[str, SampleReason] = {}
    for path in picked:
        iid = id_of.get(str(path))
        if iid is None:
            continue
        selected.append(iid)
        reasons[iid] = SampleReason(
            rank=len(selected),
            criterion=body.strategy,
            score=None,            # 见下方 note
        )

    return SampleResponse(
        selected=selected,
        reasons=reasons,
        strategy=body.strategy,
        pool_size=len(ids),
        k=len(selected),
        latency_ms=round(latency, 1),
        note=(
            "reasons.score 恒为 null：模型侧 select() 按签名只返回路径列表，"
            "不返回分值；rank 即选样顺序。要拿到真实不确定性分数，"
            "需要 m8_al 的 score_* 一并返回分值（模型侧改动）。"
        ),
    )


def _annotated_ids() -> set[str]:
    """扫 out/annotations/*/ 下已有的 .txt，得到"已经标过"的 image_id 集合。

    约定（ADDENDUM §4.3）：文件名就是 image_id（已过 safe_name）。
    """
    done: set[str] = set()
    base = config.ANNOTATION_DIR
    if not base.is_dir():
        return done
    for arm in base.iterdir():
        if not arm.is_dir():
            continue
        for f in arm.glob("*.txt"):
            done.add(safe_name(f.stem))
    return done
