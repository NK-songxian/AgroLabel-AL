"""GET /api/images —— 标注队列 / 元信息批量查询

支持两种用法：
    分页：?split=train&offset=0&limit=50&order=path
    按 id 批量取：?ids=0007,0012          （/api/sample 返回后用它落地）
    文本搜索：?q=apple                    （id / 文件名子串，不区分大小写）
"""

from __future__ import annotations

import mimetypes

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

from .. import config
from ..schemas import GtInstance, GtResponse, ImageItem, ImageListResponse

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
    q: str | None = None,
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
        split=split, class_id=class_id, ids=id_list, q=q,
        offset=offset, limit=limit, order=order,
    )
    return ImageListResponse(
        total=total, offset=offset, limit=limit,
        items=[_public(r) for r in rows],
    )


def _public(rec) -> ImageItem:
    """单条记录的对外视图。

    `AGROLABEL_HIDE_GT=1` 时把真值类清空 —— 实验期间队列不能显示"这张图属于哪类"，
    否则等于把选择题答案给标注员。注意过滤/排序仍在服务端按真值做，
    只是不再返回给浏览器。
    """
    d = rec.public()
    if config.HIDE_GT:
        d["gt_class_id"] = None
        d["gt_class_name"] = None
        d["disease"] = None
    return ImageItem(**d)


@router.get("/images/{image_id}/gt", response_model=GtResponse)
def get_image_gt(image_id: str, request: Request):
    """返回这张图的数据集真值轮廓（YOLO-seg → 归一化多边形）。

    **仅供开发/QA 对照**：后端必须设 `AGROLABEL_GT_VIEW=1` 才开，
    否则 403（默认就是关的，实验期间不会被误开）。
    真值只读，不会被写进任何标注产物。
    """
    if not config.GT_VIEW:
        raise HTTPException(
            403, "真值查看未开启：设 AGROLABEL_GT_VIEW=1 后重启后端（实验期间请勿开启）"
        )
    catalog = request.app.state.catalog
    rec = catalog.get(image_id)
    if rec is None:
        raise HTTPException(404, f"未知 image_id: {image_id}")

    # images/<split>/<stem>.jpg -> labels/<split>/<stem>.txt
    # parents: [0]=<split> [1]=images [2]=数据集根
    root = rec.path.parents[2]
    lbl = root / "labels" / rec.split / f"{rec.path.stem}.txt"
    names = (getattr(catalog, "names", None)
             or getattr(config, "CLASS_NAMES", None) or [])
    out: list[GtInstance] = []
    found = lbl.is_file()
    if found:
        try:
            with lbl.open(encoding="utf-8") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) < 7:      # class + 至少 3 个点
                        continue
                    try:
                        cid = int(parts[0])
                        nums = [float(v) for v in parts[1:]]
                    except ValueError:
                        continue
                    if len(nums) % 2:
                        nums = nums[:-1]
                    poly = [[nums[i], nums[i + 1]] for i in range(0, len(nums), 2)]
                    if len(poly) < 3:
                        continue
                    out.append(GtInstance(
                        class_id=cid,
                        class_name=(str(names[cid]) if 0 <= cid < len(names) else None),
                        polygon=poly,
                    ))
                    if len(out) >= 500:
                        break
        except OSError:
            found = False

    return GtResponse(
        image_id=image_id, found=found,
        image_width=rec.width, image_height=rec.height, instances=out,
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
