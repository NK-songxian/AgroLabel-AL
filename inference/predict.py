"""模型侧推理入口：`predict(image_path) -> list[Instance]`。

对应 `docs/HANDOFF_FRONTEND_BACKEND.md` §6 约定的签名，供后端
`/api/predict` 与 `/api/refine` 直接调用。`/api/refine` 只需把 crop
落成临时文件，再调本函数即可（坐标系相对传入图，偏移换算由后端负责，
见 `docs/INTERFACE_ADDENDUM.md` §2.6）。

返回结构与 `docs/INTERFACE.md` §1 一致::

    [{"class_id": 0, "class_name": "Apple", "score": 0.58,
      "polygon": [[x, y], ...],      # 归一化 0~1
      "bbox": [xmin, ymin, xmax, ymax]},  # 归一化 0~1
     ...]

设计要点：
- **懒加载 + 单例**：后端启动时不需要预热，第一次调用才载入权重。
- **线程锁**：ultralytics 的 `model.predict` 不是线程安全的，backend 多线程
  并发时用一把锁串行化推理，避免显存里同一块 buffer 被踩。
- **坐标归一化**：`res.masks.xy` 是**相对传入图**的像素坐标，这里除以 W/H。
  后端做 ROI refine 时必须先加 crop 偏移再归一化，别直接把这里的归一化值当原图用。

用法（自测）::

    python inference/predict.py path/to/img.jpg
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]

DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.25

_LOCK = threading.Lock()
_MODELS: dict[tuple[str, str], object] = {}


def imread_unicode(path) -> np.ndarray | None:
    """读图，兼容中文路径（cv2.imread 在 Windows 中文路径下会失败）。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def default_weights() -> Path:
    """权重查找顺序：环境变量 → weights/baseline_*.pt → runs/**/best.pt。"""
    env = os.environ.get("AGROLABEL_WEIGHTS")
    if env:
        p = Path(env)
        if p.exists():
            return p
    cands = sorted((PROJECT / "weights").glob("baseline_*.pt"))
    if cands:
        return cands[-1]
    runs = sorted((PROJECT / "runs").rglob("best.pt"))
    if runs:
        return runs[-1]
    raise FileNotFoundError(
        "找不到权重：请放 weights/baseline_*.pt，或设 AGROLABEL_WEIGHTS 指向 .pt")


def _resolve_device(device) -> str:
    if device is not None and device != "":
        return str(device)
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_model(weights: Path, device: str):
    key = (str(weights), device)
    model = _MODELS.get(key)
    if model is None:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        _MODELS[key] = model
    return model


def model_version(weights=None) -> str:
    """`文件名@sha256:前8位`，写进操作日志便于追溯是哪版权重。"""
    w = Path(weights) if weights else default_weights()
    digest = hashlib.sha256(w.read_bytes()).hexdigest()[:8]
    return f"{w.name}@sha256:{digest}"


def image_size(image_path) -> tuple[int, int]:
    bgr = imread_unicode(image_path)
    if bgr is None:
        raise FileNotFoundError(f"读不到图像：{image_path}")
    h, w = bgr.shape[:2]
    return w, h


def _class_name(names, cid: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cid, cid))
    try:
        return str(names[cid])
    except Exception:
        return str(cid)


def _result_to_instances(res, w: int, h: int) -> list[dict]:
    """ultralytics Result -> 归一化实例列表（全部坐标归一到 0~1）。"""
    out: list[dict] = []
    if res.boxes is None or len(res.boxes) == 0:
        return out

    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    xyxy = res.boxes.xyxy.cpu().numpy()
    polys = None
    if res.masks is not None and getattr(res.masks, "xy", None) is not None:
        polys = res.masks.xy

    for i in range(len(cls)):
        cid = int(cls[i])
        inst = {
            "class_id": cid,
            "class_name": _class_name(res.names, cid),
            "score": round(float(conf[i]), 4),
            "polygon": [],
            "bbox": None,
        }
        poly = polys[i] if polys is not None and i < len(polys) else None
        if poly is not None and len(poly) >= 3:
            xy = np.asarray(poly, dtype=float)
            inst["polygon"] = [
                [round(float(x) / w, 5), round(float(y) / h, 5)] for x, y in xy
            ]
            x1, y1 = xy[:, 0].min(), xy[:, 1].min()
            x2, y2 = xy[:, 0].max(), xy[:, 1].max()
        else:
            x1, y1, x2, y2 = xyxy[i]
        inst["bbox"] = [
            round(float(x1) / w, 5),
            round(float(y1) / h, 5),
            round(float(x2) / w, 5),
            round(float(y2) / h, 5),
        ]
        out.append(inst)
    return out


def _predict_bgr(bgr, weights, conf, imgsz, device):
    """对 ndarray 跑一次，返回 (instances 归一化到本图, latency_ms)。"""
    h, w = bgr.shape[:2]
    wpath = Path(weights) if weights else default_weights()
    dev = _resolve_device(device)
    model = _load_model(wpath, dev)

    t0 = time.perf_counter()
    with _LOCK:
        res = model.predict(
            bgr, imgsz=imgsz, conf=conf, device=dev, verbose=False
        )[0]
    latency = (time.perf_counter() - t0) * 1000.0
    return _result_to_instances(res, w, h), latency


def _infer(image_path, weights, conf, imgsz, device):
    """内部：跑一次，返回 (instances, w, h, latency_ms)。"""
    bgr = imread_unicode(image_path)
    if bgr is None:
        raise FileNotFoundError(f"读不到图像：{image_path}")
    instances, latency = _predict_bgr(bgr, weights, conf, imgsz, device)
    h, w = bgr.shape[:2]
    return instances, w, h, latency


def predict(
    image_path,
    weights=None,
    conf: float = DEFAULT_CONF,
    imgsz: int = DEFAULT_IMGSZ,
    device=None,
) -> list[dict]:
    """交接文档 §6 约定的签名：返回 `list[Instance]`（归一化坐标）。"""
    instances, _w, _h, _lat = _infer(image_path, weights, conf, imgsz, device)
    return instances


def predict_response(
    image_path,
    weights=None,
    conf: float = DEFAULT_CONF,
    imgsz: int = DEFAULT_IMGSZ,
    device=None,
) -> dict:
    """给 `/api/predict` 用的富响应：额外带 image_width/height、latency、model_version。"""
    instances, w, h, latency = _infer(image_path, weights, conf, imgsz, device)
    return {
        "instances": instances,
        "image_width": w,
        "image_height": h,
        "latency_ms": round(latency, 1),
        "model_version": model_version(weights),
    }


def _clamp(v, lo, hi):
    return max(lo, min(v, hi))


def _simplify(pts_px: np.ndarray, max_vertices: int) -> np.ndarray:
    """在像素坐标下用 approxPolyDP 简化到 <= max_vertices（eps 按周长比例，尺度无关）。"""
    pts = np.ascontiguousarray(pts_px.reshape(-1, 1, 2).astype(np.float32))
    peri = cv2.arcLength(pts, True)
    eps = max(0.5, 0.004 * peri)
    approx = cv2.approxPolyDP(pts, eps, True)
    while len(approx) > max_vertices and eps < peri + 1.0:
        eps *= 1.3
        approx = cv2.approxPolyDP(pts, eps, True)
    return approx.reshape(-1, 2)


def refine(
    image_path,
    point,
    radius=None,
    class_id=None,
    max_vertices: int = 64,
    weights=None,
    conf: float = DEFAULT_CONF,
    imgsz: int = DEFAULT_IMGSZ,
    device=None,
) -> dict:
    """ROI 放大重识别 —— `INTERFACE_ADDENDUM.md` §2.6 的**参考实现（可选）**。

    后端可以直接调本函数，也可以自己裁图 + 调 `predict()` 再按 §2.6 换算；
    两种方式结果应一致。

    坐标约定（与文档一致）：`point` 是**原图像素坐标**；返回的
    `polygon` / `bbox` 是**原图归一化坐标**。未检出时返回 `found=False`，
    不抛异常（这是查询成功但无结果，不是错误）。
    """
    bgr = imread_unicode(image_path)
    if bgr is None:
        raise FileNotFoundError(f"读不到图像：{image_path}")
    H, W = bgr.shape[:2]
    px, py = float(point[0]), float(point[1])
    mx = min(1024, min(W, H) // 2)
    hi = max(48, mx)

    if radius is None:
        r = int(_clamp(round(0.08 * min(W, H)), 48, 512))
    else:
        r = int(_clamp(int(radius), 48, hi))

    # 贴边时整体向内平移，保证 crop 仍是完整的 2r×2r 且一定含点击点
    x0 = int(max(0, min(px - r, W - 2 * r)))
    x1 = int(min(W, x0 + 2 * r))
    y0 = int(max(0, min(py - r, H - 2 * r)))
    y1 = int(min(H, y0 + 2 * r))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return {"found": False, "polygon": None, "bbox": None, "latency_ms": 0.0}

    crop = bgr[y0:y1, x0:x1]
    cw, ch = x1 - x0, y1 - y0

    instances, latency = _predict_bgr(crop, weights, conf, imgsz, device)
    if class_id is not None:
        instances = [i for i in instances if i["class_id"] == int(class_id)]

    found = None
    if instances:
        lx, ly = px - x0, py - y0  # 点击点在 crop 内的像素坐标
        best_contain = None
        best_centroid = None
        best_cd = 1e18
        for inst in instances:
            poly = inst.get("polygon") or []
            if len(poly) < 3:
                continue
            pts = np.array([[x * cw, y * ch] for x, y in poly], np.float32)
            inside = cv2.pointPolygonTest(
                pts.reshape(-1, 1, 2), (lx, ly), False
            ) >= 0
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            cd = (cx - lx) ** 2 + (cy - ly) ** 2
            if inside and (best_contain is None or inst["score"] > best_contain["score"]):
                best_contain = inst
            if cd < best_cd:
                best_cd, best_centroid = cd, inst
        found = best_contain or best_centroid

    if found is None:
        return {"found": False, "polygon": None, "bbox": None,
                "latency_ms": round(latency, 1)}

    poly = found.get("polygon") or []
    pts = np.array([[x * cw, y * ch] for x, y in poly], np.float32)
    raw_vertices = int(len(pts))
    raw_area = float(abs(cv2.contourArea(pts))) if raw_vertices >= 3 else 0.0
    approx = _simplify(pts, max_vertices) if raw_vertices > max_vertices else pts

    # crop 像素 -> 原图归一化
    poly_norm = [
        [round(float((x0 + q[0]) / W), 5), round(float((y0 + q[1]) / H), 5)]
        for q in approx
    ]
    xs, ys = approx[:, 0], approx[:, 1]
    bbox = [
        round(float((x0 + xs.min()) / W), 5),
        round(float((y0 + ys.min()) / H), 5),
        round(float((x0 + xs.max()) / W), 5),
        round(float((y0 + ys.max()) / H), 5),
    ]
    coverage = raw_area / float(cw * ch) if cw and ch else 0.0

    return {
        "found": True,
        "class_id": found["class_id"],
        "class_name": found["class_name"],
        "score": found["score"],
        "polygon": poly_norm,
        "bbox": bbox,
        "coverage": round(coverage, 4),
        "raw_vertices": raw_vertices,
        "latency_ms": round(latency, 1),
    }


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="模型侧推理入口自测")
    ap.add_argument("image", help="图像路径")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--device", default="cpu", help="默认 cpu，不抢占训练卡")
    ap.add_argument("--point", nargs=2, type=float, default=None,
                    help="给了就改跑 refine：原图像素坐标 x y")
    ap.add_argument("--radius", type=int, default=None)
    ap.add_argument("--class-id", type=int, default=None)
    ap.add_argument("--max-vertices", type=int, default=64)
    args = ap.parse_args()

    if args.point:
        import json

        r = refine(args.image, args.point, radius=args.radius,
                   class_id=args.class_id, max_vertices=args.max_vertices,
                   weights=args.weights, conf=args.conf,
                   imgsz=args.imgsz, device=args.device)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return

    r = predict_response(
        args.image,
        weights=args.weights,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    print(f"image={args.image}")
    print(f"size={r['image_width']}x{r['image_height']}  "
          f"latency={r['latency_ms']} ms  version={r['model_version']}")
    print(f"instances={len(r['instances'])}")
    for inst in r["instances"][:5]:
        print(f"  cls={inst['class_id']}({inst['class_name']}) "
              f"score={inst['score']} ptrs={len(inst['polygon'])} "
              f"bbox={inst['bbox']}")


if __name__ == "__main__":
    _main()
