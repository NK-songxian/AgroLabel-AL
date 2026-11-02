"""POST /api/edit-cost —— 预标注 vs 真值的编辑代价

契约：docs/INTERFACE.md §2 + ADDENDUM §3.3

口径不在这里实现：直接调 `src/m6_editcost.py::edit_ops()`，
保证"接口算出来的数字"和"离线跑 m6 得到的数字"是同一套。

本路由只负责一件事：契约收的是**多边形**，而 `edit_ops` 吃的是**掩码**，
所以要把归一化多边形栅格化。两边用同一块画布，IoU 才有意义。
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from .. import config
from ..schemas import EditCostRequest, EditCostResponse, InInstance

router = APIRouter(prefix="/api", tags=["editcost"])

BREAKDOWN_KEYS = ("新画", "删实例", "改类别", "拖边界", "标为无目标")
DEFAULT_CANVAS = 1024


def _pick_canvas(req: EditCostRequest) -> tuple[int, int]:
    """栅格化画布尺寸。给了真实宽高就用，否则用 1024×1024 方画布。

    方画布不会让 IoU 失真 —— gt 和 pred 被同样的变换压扁，
    只要两边一致，相对关系就保住了。给真实宽高只是让结果更贴近人眼。
    """
    w = req.image_width or DEFAULT_CANVAS
    h = req.image_height or DEFAULT_CANVAS
    if w > 4096 or h > 4096:
        scale = 4096 / max(w, h)
        w, h = int(w * scale), int(h * scale)
    return max(8, int(w)), max(8, int(h))


def _to_masks(instances: list[InInstance], w: int, h: int):
    """[InInstance] -> [(class_id, mask)]，mask 为 uint8 的 0/1，与 m6 一致。"""
    out = []
    for inst in instances:
        pts = np.array([[x * w, y * h] for x, y in inst.polygon], np.float32)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        if m.sum() == 0:          # 退化成线/点的多边形，m6 也会跳过
            continue
        out.append((int(inst.class_id), m))
    return out


def _load_edit_ops():
    src = str(config.PROJECT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from m6_editcost import edit_ops  # noqa: PLC0415
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(
            501, f"src/m6_editcost.py 导入失败：{type(e).__name__}: {e}"
        ) from e
    return edit_ops


@router.post("/edit-cost", response_model=EditCostResponse)
def edit_cost(body: EditCostRequest):
    w, h = _pick_canvas(body)

    gt = _to_masks(body.gt_instances, w, h)
    pred = _to_masks(body.pred_instances, w, h)

    if len(gt) != len(body.gt_instances):
        raise HTTPException(
            422,
            f"有 {len(body.gt_instances) - len(gt)} 个 gt 多边形栅格化后面积为 0，"
            f"检查顶点是否退化或坐标是否在 [0,1]",
        )

    edit_ops = _load_edit_ops()
    try:
        count, counter, ious = edit_ops(gt, pred)
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(500, f"edit_ops 执行失败：{type(e).__name__}: {e}")

    breakdown = {k: int(counter.get(k, 0)) for k in BREAKDOWN_KEYS}
    unknown = {k: int(v) for k, v in counter.items() if k not in BREAKDOWN_KEYS}

    return EditCostResponse(
        edit_count=int(count),
        breakdown=breakdown,
        matched_ious=[round(float(v), 4) for v in ious],
        canvas=[w, h],
        note=(
            "口径同 src/m6_editcost.py：同类内贪心 IoU 匹配，"
            "IOU_MATCH=0.5 视为同一实例，IoU<0.75 记为拖边界。"
            + (f" 另出现非常规键 {unknown}，请检查。" if unknown else "")
        ),
    )
