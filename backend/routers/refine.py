"""POST /api/refine —— 点提示：裁 ROI 放大重识别

契约：docs/INTERFACE_ADDENDUM.md §2.6

算法在模型侧 `inference/predict.py::refine()`（它已经是按 §2.6 写的参考实现，
连"输入像素 / 输出归一化"这个不对称都处理好了）。本路由只做 image_id -> 路径。

为什么这个功能值钱：基线 Mask Recall 只有 0.32，一张图平均 7.2 个病斑要人补 5 个；
病斑面积中位仅整图 0.3%，逐点描一个多边形要 22 次点击，而点一下让模型在放大后的
局部再看一眼只需 8.8ms。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from .. import config
from ..schemas import RefineResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["refine"])


class RefineRequest(BaseModel):
    image_id: str = Field(..., min_length=1, max_length=256)
    point: list[float] = Field(..., min_length=2, max_length=2,
                               description="原图像素坐标 [x, y]，不是归一化")
    radius: int | None = Field(default=None, ge=48, le=1024)
    class_id: int | None = Field(default=None, ge=0)
    max_vertices: int = Field(default=64, ge=16, le=256)


@router.post("/refine", response_model=RefineResponse)
def refine(body: RefineRequest, request: Request):
    catalog = request.app.state.catalog

    rec = catalog.get(body.image_id)
    if rec is None:
        raise HTTPException(404, f"未知 image_id: {body.image_id}")
    if not rec.path.is_file():
        raise HTTPException(410, f"文件已不在磁盘上: {rec.path.name}")

    try:
        from inference.predict import refine as model_refine
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(
            501, f"模型侧 inference/predict.py 导入失败：{type(e).__name__}: {e}"
        )

    try:
        out = model_refine(
            rec.path,
            body.point,
            radius=body.radius,
            class_id=body.class_id,
            max_vertices=body.max_vertices,
            weights=str(config.WEIGHTS),
            conf=config.CONF,
            imgsz=config.IMGSZ,
            device=config.resolve_device(),
        )
    except FileNotFoundError as e:
        raise HTTPException(410, str(e))
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(500, f"refine 执行失败：{type(e).__name__}: {e}")

    # found=False 是"查询成功但没结果"，不是错误 —— 前端要靠它走手画兜底
    return RefineResponse(**out)
