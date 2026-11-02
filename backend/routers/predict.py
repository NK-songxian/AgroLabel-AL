"""POST /api/predict —— 单张预标注

契约（INTERFACE.md §2 + ADDENDUM §3.1）：
    入参  {image_id} 或 {image_base64}
    出参  {instances:[{class_id,class_name,score,polygon,bbox}], latency_ms}

比原契约多返回三个字段（向后兼容的新增）：
    image_width / image_height  前端把归一化多边形还原成像素必需
    model_version               写进操作日志，能追溯是哪版权重
"""

from __future__ import annotations

import base64
import binascii
import uuid

from fastapi import APIRouter, HTTPException, Request

from .. import config
from ..model_bridge.predictor import utcnow
from ..schemas import PredictRequest, PredictResponse

router = APIRouter(prefix="/api", tags=["predict"])

MAX_B64_BYTES = 512 * 1024
TMP_DIR = config.OUT_DIR / "tmp"


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    predictor = request.app.state.predictor
    catalog = request.app.state.catalog

    if body.image_id:
        rec = catalog.get(body.image_id)
        if rec is None:
            raise HTTPException(404, f"未知 image_id: {body.image_id}")
        if not rec.path.is_file():
            raise HTTPException(410, f"文件已不在磁盘上: {rec.path.name}")
        path, image_id, cleanup = rec.path, body.image_id, None
    else:
        raw = _decode_b64(body.image_base64)
        if len(raw) > MAX_B64_BYTES:
            raise HTTPException(
                413,
                f"image_base64 解码后 {len(raw) // 1024} KB，超过 {MAX_B64_BYTES // 1024} KB；"
                f"大图请改用 image_id",
            )
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        tmp = TMP_DIR / f"b64_{uuid.uuid4().hex}.jpg"
        tmp.write_bytes(raw)
        path, image_id, cleanup = tmp, f"b64_{tmp.stem}", tmp

    try:
        instances, latency_ms, (w, h) = predictor.predict_image(path)

        # 契约 §4.2：预标注结果必须落盘，供 /api/annotations 取 pred_snapshot
        predictor.cache_pred(image_id, {
            "image_id": image_id,
            "instances": instances,
            "image_width": w,
            "image_height": h,
            "model_version": predictor.model_version,
            "pred_ts": utcnow(),
        })

        return PredictResponse(
            instances=instances,
            latency_ms=round(latency_ms, 1),
            image_width=w,
            image_height=h,
            model_version=predictor.model_version,
        )
    finally:
        if cleanup is not None:
            cleanup.unlink(missing_ok=True)


def _decode_b64(s: str | None) -> bytes:
    if not s:
        raise HTTPException(400, "image_base64 为空")
    if "," in s[:64]:          # 允许 data:image/jpeg;base64,xxx 形式
        s = s.split(",", 1)[1]
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(400, f"image_base64 不是合法的 base64: {e}")
