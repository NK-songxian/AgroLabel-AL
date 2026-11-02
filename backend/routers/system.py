"""GET /api/health 与 GET /api/classes —— 系统自检与类别字典。"""

from __future__ import annotations

from fastapi import APIRouter, Request

from .. import config
from ..schemas import ClassItem, ClassesResponse, DatasetInfo, HealthResponse, ModelInfo

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    catalog = request.app.state.catalog
    predictor = request.app.state.predictor

    return HealthResponse(
        status="ok",
        model=config.WEIGHTS.name,
        classes=config.NC,
        dataset=DatasetInfo(
            mode=catalog.mode,
            note=config.DATA.note,
            total=len(catalog),
            splits=catalog.splits(),
            classes_present=len(catalog.class_counts()),
        ),
        model_info=ModelInfo(**predictor.describe()),
    )


@router.get("/classes", response_model=ClassesResponse)
def classes():
    """34 类字典。数据源固定为 configs 的 names，前端不要另写一份。"""
    return ClassesResponse(
        nc=config.NC,
        classes=[ClassItem(class_id=i, class_name=n) for i, n in enumerate(config.CLASS_NAMES)],
    )
