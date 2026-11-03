"""POST /api/annotations —— 提交修正结果（标注闭环的落点）

契约：docs/INTERFACE_ADDENDUM.md §2.4 / §4.1 / §4.2 / §4.3

落盘（全部在 out/ 下，绝不写 datasets/）：
    out/annotations/<arm>/<image_id>.txt    YOLO seg：class_id x1 y1 x2 y2 ...（归一化）
    out/annotations/<arm>/<image_id>.json   元数据：label_state / pred_snapshot / 耗时 / 操作员

pred_snapshot **不由前端传**：从 out/pred_cache/<image_id>.json 读（§4.2）。
缓存缺失时记为 []，且**提交照常成功** —— 标注是主数据，快照只是旁证。
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query, Request

from .. import config
from ..schemas import (
    AnnotateRequest,
    AnnotateResponse,
    AnnotationReadResponse,
    SavedInstance,
)
from ..util import atomic_write_text, now_stamp, safe_name

router = APIRouter(prefix="/api", tags=["annotations"])

COORD_TOL = 1e-6          # 归一化坐标的容差
MAX_VERTICES = 512        # 单实例顶点上限，防止伪造超大多边形撑爆磁盘


def _resolve_arm(arm: str | None) -> str:
    """解析并校验实验臂。

    为什么要白名单：`out/annotations/<arm>/` 是按臂分目录的，臂名写错一个字母
    （比如 "assisted " 带空格、"Assisted" 大写）就会静默多出一个目录，
    三臂对照的分母就错了，而且事后极难发现。未知臂名一律 422。
    """
    name = safe_name(arm or config.DEFAULT_ARM, fallback=config.DEFAULT_ARM)
    if name not in config.VALID_ARMS:
        raise HTTPException(
            422,
            f"未知实验臂 {name!r}，可选 {list(config.VALID_ARMS)}。"
            f"臂名决定标注落到 out/annotations/<arm>/ 的哪个目录，写错会毁掉对照实验",
        )
    return name


@router.get("/annotations/{image_id}", response_model=AnnotationReadResponse)
def read_annotation(
    image_id: str,
    arm: str | None = Query(
        default=None,
        description="实验臂；缺省取 AGROLABEL_ARM，再缺省 assisted",
    ),
):
    """读回已保存的标注。

    没有它，刷新页面就会把标注员的工作全部丢掉 —— 前端只能重新调 /api/predict，
    而预标注不是他改完的结果。ADDENDUM §2.7 就是为这个场景留的。
    """
    arm_name = _resolve_arm(arm)
    image_id = safe_name(image_id, fallback="unnamed")

    arm_dir = config.ANNOTATION_DIR / arm_name
    label_path = arm_dir / f"{image_id}.txt"
    meta_path = arm_dir / f"{image_id}.json"
    if not label_path.is_file():
        # 200 + found=false：这不是错误，是"还没标过"。
        # 用 404 会让浏览器控制台每张图刷一条红色错误，演示时很难看。
        return AnnotationReadResponse(
            found=False, image_id=image_id, arm=arm_name, instances=[], n_instances=0
        )

    instances: list[SavedInstance] = []
    text = label_path.read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 7 or (len(parts) - 1) % 2:
            raise HTTPException(
                422,
                f"{label_path.name} 第 {line_no} 行不是合法的 YOLO seg 行"
                f"（应为 class_id + 成对的 x y，实际 {len(parts)} 个字段）",
            )
        cid = int(parts[0])
        vals = [float(v) for v in parts[1:]]
        poly = [[vals[i], vals[i + 1]] for i in range(0, len(vals), 2)]
        instances.append(
            SavedInstance(
                class_id=cid,
                class_name=config.CLASS_NAMES[cid]
                if 0 <= cid < len(config.CLASS_NAMES) else "",
                polygon=poly,
            )
        )

    meta: dict = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    return AnnotationReadResponse(
        found=True,
        image_id=image_id,
        arm=arm_name,
        label_state=meta.get("label_state", "labeled"),
        instances=instances,
        n_instances=len(instances),
        elapsed_ms=meta.get("elapsed_ms"),
        saved_at=meta.get("ts"),
        pred_cached=bool(meta.get("pred_cached")),
    )


@router.post("/annotations", response_model=AnnotateResponse)
def submit_annotation(
    body: AnnotateRequest,
    request: Request,
    arm: str | None = Query(
        default=None,
        description="实验臂，落到 out/annotations/<arm>/。缺省取 AGROLABEL_ARM，再缺省 assisted",
    ),
):
    predictor = request.app.state.predictor

    arm_name = _resolve_arm(arm)
    image_id = safe_name(body.image_id, fallback="unnamed")

    _validate_instances(body.instances)

    # ---- 取 pred_snapshot（§4.2：后端缓存，前端不传）
    cached = predictor.read_cached_pred(body.image_id)
    pred_snapshot = (cached or {}).get("instances") or []
    pred_cached = bool(cached)
    model_version = (cached or {}).get("model_version")

    saved_at = now_stamp()
    arm_dir = config.ANNOTATION_DIR / arm_name
    label_path = arm_dir / f"{image_id}.txt"
    meta_path = arm_dir / f"{image_id}.json"

    # ---- YOLO seg txt：每个实例一行 "class_id x1 y1 x2 y2 ..."
    #      no_target 也要落一个空文件 —— 空文件和"没标"必须能区分，见 §4.1
    lines = []
    for inst in body.instances:
        coords = " ".join(f"{v:.6f}" for pt in inst.polygon for v in pt)
        lines.append(f"{inst.class_id} {coords}")
    atomic_write_text(label_path, ("\n".join(lines) + "\n") if lines else "")

    # ---- 元数据 sidecar
    meta = {
        "image_id": body.image_id,
        "label_state": body.label_state,
        "arm": arm_name,
        "n_instances": len(body.instances),
        "pred_snapshot": pred_snapshot,
        "pred_cached": pred_cached,
        "model_version": model_version,
        "elapsed_ms": body.elapsed_ms,
        "operator": body.operator,
        "session_id": body.session_id,
        "ts": saved_at,
    }
    atomic_write_text(meta_path, json.dumps(meta, ensure_ascii=False, indent=2))

    return AnnotateResponse(
        ok=True,
        image_id=body.image_id,
        n_instances=len(body.instances),
        label_state=body.label_state,
        arm=arm_name,
        pred_cached=pred_cached,
        label_path=str(label_path.relative_to(config.PROJECT)).replace("\\", "/"),
        saved_at=saved_at,
    )


def _validate_instances(instances) -> None:
    """几何与类别的合法性。宁可 422 打回，也不要写出坏标签污染 m6 的统计。"""
    nc = config.NC
    for i, inst in enumerate(instances):
        if not (0 <= inst.class_id < nc):
            raise HTTPException(422, f"instances[{i}].class_id={inst.class_id} 越界，应在 [0,{nc})")
        n = len(inst.polygon)
        if n < 3:
            raise HTTPException(422, f"instances[{i}] 只有 {n} 个顶点，多边形至少要 3 个")
        if n > MAX_VERTICES:
            raise HTTPException(422, f"instances[{i}] 有 {n} 个顶点，超过上限 {MAX_VERTICES}")
        for j, pt in enumerate(inst.polygon):
            if len(pt) != 2:
                raise HTTPException(422, f"instances[{i}].polygon[{j}] 应为 [x,y]，收到 {pt}")
            x, y = pt
            if not (-COORD_TOL <= x <= 1 + COORD_TOL and -COORD_TOL <= y <= 1 + COORD_TOL):
                raise HTTPException(
                    422,
                    f"instances[{i}].polygon[{j}]=({x},{y}) 超出 [0,1]；"
                    f"契约要求归一化坐标，像素坐标请先除以宽高",
                )
