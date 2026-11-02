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


# ---------------------------------------------------------------- 提交标注

class AnnotateRequest(BaseModel):
    """POST /api/annotations 的请求体（ADDENDUM §2.4）。

    注意：**没有 pred_snapshot** —— 契约 §4.2 决定由后端从 out/pred_cache 读。
    """
    image_id: str = Field(..., min_length=1, max_length=256)
    label_state: Literal["labeled", "no_target"]
    instances: list[InInstance] = Field(default_factory=list)
    elapsed_ms: int = Field(default=0, ge=0)
    operator: str | None = Field(default=None, max_length=64)
    session_id: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _no_target_must_be_empty(self):
        # 契约 §4.1：no_target 时 instances 必须是 []
        if self.label_state == "no_target" and self.instances:
            raise ValueError("label_state=no_target 时 instances 必须为空；"
                             "若确实标出了目标请用 label_state=labeled")
        return self


class AnnotateResponse(BaseModel):
    ok: bool
    image_id: str
    n_instances: int
    label_state: str
    arm: str
    pred_cached: bool
    label_path: str
    saved_at: str


# ---------------------------------------------------------------- 操作日志

class OpEventIn(BaseModel):
    ts: str = Field(..., min_length=1, max_length=32)
    image: str = Field(..., min_length=1, max_length=256)
    op: str = Field(..., min_length=1, max_length=16)
    instance: int = -1
    detail: str = ""
    n_ops_on_image: int = 0
    api_ms: str = ""


class OpsRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    events: list[OpEventIn] = Field(..., min_length=1, max_length=500)


class OpsResponse(BaseModel):
    ok: bool
    appended: int
    rows_total: int
    path: str
