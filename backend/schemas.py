"""请求 / 响应模型。字段名严格对齐 docs/INTERFACE.md 与 docs/INTERFACE_ADDENDUM.md。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------- 实例

class OutInstance(BaseModel):
    """模型输出的实例。坐标一律归一化到 [0,1]。"""
    class_id: int
    class_name: str
    score: float
    polygon: list[list[float]] = Field(..., description="归一化多边形顶点 [[x,y], ...]")
    bbox: list[float] = Field(..., description="归一化 [xmin, ymin, xmax, ymax]")


class InInstance(BaseModel):
    """人工提交的实例。score 可为空（人改过的不带模型置信度）。"""
    class_id: int
    class_name: str | None = None
    score: float | None = None
    polygon: list[list[float]]
    bbox: list[float] | None = None


# ---------------------------------------------------------------- predict

class PredictRequest(BaseModel):
    image_id: str | None = None
    image_base64: str | None = Field(
        default=None,
        description="仅用于 ≤512KB 的小图；优先用 image_id，服务端需额外解码，占 300ms 预算",
    )

    @model_validator(mode="after")
    def _exactly_one(self):
        has_id = bool(self.image_id)
        has_b64 = bool(self.image_base64)
        if has_id == has_b64:
            raise ValueError("image_id 与 image_base64 必须二选一")
        return self


class PredictResponse(BaseModel):
    instances: list[OutInstance]
    latency_ms: float
    image_width: int
    image_height: int
    model_version: str


# ---------------------------------------------------------------- images

class ImageItem(BaseModel):
    image_id: str
    file_name: str
    split: str
    width: int
    height: int
    gt_class_id: int | None = None
    gt_class_name: str | None = None
    disease: str | None = None


class ImageListResponse(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[ImageItem]


class ClassItem(BaseModel):
    class_id: int
    class_name: str


class ClassesResponse(BaseModel):
    nc: int
    classes: list[ClassItem]


# ---------------------------------------------------------------- health

class DatasetInfo(BaseModel):
    mode: Literal["manifest", "demo"]
    note: str
    total: int
    splits: list[dict]
    classes_present: int


class ModelInfo(BaseModel):
    weights: str
    model_version: str
    device: str
    imgsz: int
    conf: float
    nc: int
    load_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model: str
    classes: int
    dataset: DatasetInfo
    model_info: ModelInfo
