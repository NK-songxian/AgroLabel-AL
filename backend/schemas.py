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
    mode: Literal["manifest", "yolo", "demo"]
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
    # 打开图 -> 落盘的时长（含看图/熟悉）。手册 §08 用 elapsed_ms（首个动作起算），
    # 这个字段保留另一套口径，方便报告里同时给出"纯操作时间"和"总时长"。
    elapsed_from_open_ms: int | None = Field(default=None, ge=0)
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


class SavedInstance(BaseModel):
    class_id: int
    class_name: str = ""
    polygon: list[list[float]]


class AnnotationReadResponse(BaseModel):
    """GET /api/annotations/{image_id} —— 读回已保存的标注（续标/刷新不丢数据）

    没标过时返回 **200 + found=false**，不是 404 —— "这张标过没有"是高频探询，
    用 404 会让浏览器控制台每张图刷一条红色错误，演示时很难看。
    404 只留给"臂名非法"这类真正的错误。
    """
    found: bool
    image_id: str
    arm: str
    label_state: Literal["labeled", "no_target"] | None = None
    instances: list[SavedInstance] = []
    n_instances: int = 0
    elapsed_ms: int | None = None
    saved_at: str | None = None
    pred_cached: bool = False


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


# ---------------------------------------------------------------- 主动学习选样

class SampleRequest(BaseModel):
    """POST /api/sample（INTERFACE.md §2）。

    `pool` 省略时用整个 catalog，并默认剔除已标注过的图 —— 这样"智能选样"
    反复点也不会一直推荐同一批。
    """
    pool: list[str] | None = None
    k: int = Field(default=10, ge=1, le=200)
    strategy: Literal["random", "entropy", "tta", "coreset"] = "entropy"
    exclude_annotated: bool = True
    seed: int = 0


class SampleReason(BaseModel):
    rank: int
    criterion: str
    # 模型侧真实打分（entropy=二元熵0-1 / tta=不一致度0-1 / coreset=描述子距离；
    # random 恒为 null——随机对照没有分值）。
    score: float | None = None


class SampleResponse(BaseModel):
    selected: list[str]
    reasons: dict[str, SampleReason]
    strategy: str
    pool_size: int
    k: int
    latency_ms: float
    note: str


# ---------------------------------------------------------------- 点提示

class RefineResponse(BaseModel):
    found: bool
    class_id: int | None = None
    class_name: str | None = None
    score: float | None = None
    polygon: list[list[float]] | None = None
    bbox: list[float] | None = None
    coverage: float | None = None
    raw_vertices: int | None = None
    latency_ms: float
    # 识别出的类别与请求的类别不一致（局部放大后模型常判错物种）。
    # 前端据此提示"识别为 X，可在右栏改类"，而不是默默插错。
    class_mismatch: bool = False
    # 实际使用的 ROI 半径，便于排查"为什么没认出来"
    radius_used: int | None = None


# ---------------------------------------------------------------- 编辑代价

class EditCostRequest(BaseModel):
    """POST /api/edit-cost（INTERFACE.md §2 + ADDENDUM §3.3）。

    只要求 pred_instances / gt_instances，与契约一致。
    额外接受可选的画布尺寸：内部要把多边形栅格化成掩码才能复用
    `src/m6_editcost.py::edit_ops`，给真实宽高比灌一个方画布更准。
    """
    pred_instances: list[InInstance] = Field(default_factory=list)
    gt_instances: list[InInstance] = Field(default_factory=list)
    image_width: int | None = Field(default=None, gt=0, le=20000)
    image_height: int | None = Field(default=None, gt=0, le=20000)


class EditCostResponse(BaseModel):
    edit_count: int
    breakdown: dict[str, int]
    matched_ious: list[float]
    canvas: list[int]
    note: str
