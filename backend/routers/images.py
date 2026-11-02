"""GET /api/images —— 标注队列 / 元信息批量查询

支持两种用法：
    分页：?split=train&offset=0&limit=50&order=path
    按 id 批量取：?ids=0007,0012          （/api/sample 返回后用它落地）
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

from .. import config
from ..schemas import ImageItem, ImageListResponse

router = APIRouter(prefix="/api", tags=["images"])

MAX_LIMIT = 200
_IMG_EXT = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".bmp": "image/bmp", ".webp": "image/webp"}


@router.get("/images", response_model=ImageListResponse)
def list_images(
    request: Request,
    split: str | None = None,
    class_id: int | None = None,
    ids: str | None = None,
    offset: int = 0,
    limit: int = 50,
    order: str = "path",
):
    catalog = request.app.state.catalog

    limit = max(1, min(MAX_LIMIT, limit))
    offset = max(0, offset)

    id_list = [s.strip() for s in ids.split(",") if s.strip()] if ids else None
    if id_list is not None and len(id_list) > MAX_LIMIT:
        raise HTTPException(400, f"ids 最多 {MAX_LIMIT} 个，收到 {len(id_list)}")

    # 契约 §2.1 原本还写了 score_desc，但它需要"每张图的预标注分数"这份索引；
    # 现在只有调过 /api/predict 的图才有 out/pred_cache，覆盖不全，
    # 排出来会误导人。所以先只认 path / class，其余明确报错而不是静默忽略。
    if order not in ("path", "class"):
        raise HTTPException(
            400,
            f"order={order} 不支持，可选 path / class。"
            f"score_desc 需要预标注分数索引（目前仅 /api/predict 过的图有），排在 P3",
        )

    rows, total = catalog.query(
        split=split, class_id=class_id, ids=id_list,
        offset=offset, limit=limit, order=order,
    )
    return ImageListResponse(
        total=total, offset=offset, limit=limit,
        items=[ImageItem(**r.public()) for r in rows],
    )


@router.get("/images/{image_id}/file")
def get_image_file(image_id: str, request: Request):
    """返回原图二进制。

    需要 ETag + 304（前端大量复用同一张图），并支持 Range（大图与断点）。
    这里手写而不是直接丢给 FileResponse —— 需要可控的 ETag 与 206 语义。
    """
    catalog = request.app.state.catalog
    rec = catalog.get(image_id)
    if rec is None:
        raise HTTPException(404, f"未知 image_id: {image_id}")
    if not rec.path.is_file():
        raise HTTPException(410, f"文件已不在磁盘上: {rec.path.name}")

    st = rec.path.stat()
    etag = f'"{st.st_size:x}-{int(st.st_mtime):x}"'
    media = _IMG_EXT.get(rec.path.suffix.lower()) or \
        (mimetypes.guess_type(rec.path.name)[0] or "application/octet-stream")

    common = {
        "ETag": etag,
        "Cache-Control": "public, max-age=31536000, immutable",
        "Accept-Ranges": "bytes",
    }

    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=common)

    rng = request.headers.get("range")
    if rng and rng.startswith("bytes="):
        try:
            spec = rng[6:].split(",")[0].strip()
            s_str, _, e_str = spec.partition("-")
            if s_str == "":
                # bytes=-N 表示最后 N 字节
                n = int(e_str)
                start, end = max(0, st.st_size - n), st.st_size - 1
            else:
                start = int(s_str)
                end = int(e_str) if e_str else st.st_size - 1
                end = min(end, st.st_size - 1)
            if start > end or start >= st.st_size:
                return Response(
                    status_code=416,
                    headers={**common, "Content-Range": f"bytes */{st.st_size}"},
                )
            length = end - start + 1
            with open(rec.path, "rb") as f:
                f.seek(start)
                chunk = f.read(length)
            return Response(
                content=chunk, status_code=206, media_type=media,
                headers={
                    **common,
                    "Content-Range": f"bytes {start}-{end}/{st.st_size}",
                    "Content-Length": str(length),
                },
            )
        except (ValueError, OSError):
            pass  # Range 头畸形就退化成整文件

    return FileResponse(rec.path, media_type=media, headers=common)
