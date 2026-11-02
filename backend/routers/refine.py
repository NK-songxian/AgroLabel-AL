"""POST /api/refine —— 点提示：裁 ROI 放大重识别

契约：docs/INTERFACE_ADDENDUM.md §2.6

算法在模型侧 `inference/predict.py::refine()`（按 §2.6 写的参考实现，
连"输入像素 / 输出归一化"这个不对称都处理好了）。本路由只做三件事：

  1. image_id -> 路径
  2. **选对 ROI 半径与置信度**（见下）
  3. **要求返回的多边形真的包住点击点**，否则视为没认出来

为什么第 2、3 步是必要的（实测数据，4 张 demo 图 × 5 个病斑点 + 5 个空白点）：

    conf  radius  class_id    包住(有用)  返回但不是你点的  没认出来
    0.25  48(默认)  限定        12%          6%            81%   ← 模型侧默认值
    0.25  96       限定        38%         19%            44%
    0.10  96      不限定        75%         25%             0%

两个原因叠加导致默认配置几乎不可用：

  - **ROI 太小**：默认半径 = 0.08·min(W,H)，302x225 的图上只有 48px，
    即 96x96 的窗口被 ultralytics 放大到 640（6.7 倍）。病斑尺寸远超训练分布，
    模型在这个尺度上基本看不见。
  - **类别硬过滤**：放大后模型常把甘蓝判成小麦，一过滤就空了。

  另外 refine() 的 best_centroid 兜底没有距离限制，不限定类别时几乎总能
  "找到点什么"，但空白点也有 85~95% 会返回一个并不包含点击点的多边形。
  所以必须自己校验包含关系，不能只信 found。

策略：先按给定类别找；找不到就**不限定类别再找一次**，但一律要求多边形包住
点击点。类别不一致时返回 class_mismatch=true，由前端提示"识别为 X，可在右栏改类"。
"""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .. import config
from ..schemas import RefineResponse

router = APIRouter(prefix="/api", tags=["refine"])

# ROI 半径：默认取短边的一半，夹在 [96, 512]。
# 96 是实测下限 —— 再小就成了"放大后认不出"（见文件头表格）。
MIN_RADIUS = 96
MAX_RADIUS = 512

# 局部重识别用更低的置信度：ROI 被放大后，模型对同一病斑的分数普遍偏低
REFINE_CONF = 0.10


class RefineRequest(BaseModel):
    image_id: str = Field(..., min_length=1, max_length=256)
    point: list[float] = Field(..., min_length=2, max_length=2,
                               description="原图像素坐标 [x, y]，不是归一化")
    radius: int | None = Field(default=None, ge=48, le=1024)
    class_id: int | None = Field(default=None, ge=0)
    max_vertices: int = Field(default=64, ge=16, le=256)


def _default_radius(w: int, h: int) -> int:
    return int(max(MIN_RADIUS, min(MAX_RADIUS, round(0.5 * min(w, h)))))


def _contains(polygon, px: float, py: float, w: int, h: int) -> bool:
    """归一化多边形是否包住像素点。refine() 的兜底会返回远处的目标，必须自己验。"""
    if not polygon or len(polygon) < 3:
        return False
    pts = np.array([[x * w, y * h] for x, y in polygon], np.float32)
    return cv2.pointPolygonTest(pts.reshape(-1, 1, 2), (px, py), False) >= 0


@router.post("/refine", response_model=RefineResponse)
def refine(body: RefineRequest, request: Request):
    catalog = request.app.state.catalog

    rec = catalog.get(body.image_id)
    if rec is None:
        raise HTTPException(404, f"未知 image_id: {body.image_id}")
    if not rec.path.is_file():
        raise HTTPException(410, f"文件已不在磁盘上: {rec.path.name}")

    try:
        from inference.predict import image_size, refine as model_refine
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(
            501, f"模型侧 inference/predict.py 导入失败：{type(e).__name__}: {e}"
        )

    try:
        w, h = image_size(rec.path)
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(410, f"读不到图像尺寸：{e}")

    radius = body.radius if body.radius is not None else _default_radius(w, h)
    px, py = float(body.point[0]), float(body.point[1])

    def call(cid):
        return model_refine(
            rec.path,
            [px, py],
            radius=radius,
            class_id=cid,
            max_vertices=body.max_vertices,
            weights=str(config.WEIGHTS),
            conf=REFINE_CONF,
            imgsz=config.IMGSZ,
            device=config.resolve_device(),
        )

    try:
        # 第一遍：按调用方给的类别（前端传的是图像 gt 类）
        out = call(body.class_id)
        mismatch = False
        if body.class_id is not None and not (
            out["found"] and _contains(out.get("polygon"), px, py, w, h)
        ):
            # 第二遍：不限定类别。放大后模型常判错物种，硬过滤会把正确答案丢掉。
            alt = call(None)
            if alt["found"] and _contains(alt.get("polygon"), px, py, w, h):
                out, mismatch = alt, True
            elif out["found"] and not _contains(out.get("polygon"), px, py, w, h):
                out = {**out, "found": False, "polygon": None, "bbox": None}
    except FileNotFoundError as e:
        raise HTTPException(410, str(e))
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(500, f"refine 执行失败：{type(e).__name__}: {e}")

    # 最终仍要求包住点击点：refine() 的 best_centroid 兜底会返回"附近随便一个"，
    # 不校验的话，点在空白处也会插出一个莫名其妙的多边形
    if out["found"] and not _contains(out.get("polygon"), px, py, w, h):
        out = {**out, "found": False, "polygon": None, "bbox": None}

    # found=False 是"查询成功但没结果"，不是错误 —— 前端要靠它走手画兜底
    return RefineResponse(**out, class_mismatch=mismatch, radius_used=radius)
