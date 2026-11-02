"""模型封装：进程内只加载一次 YOLO，提供 predict 与 pred 缓存落盘。

两个实现上的硬性要求（都在本机实测踩过）：
  1. GPU 上 ultralytics 返回的是 CUDA tensor，取 conf/cls/masks 之前必须 .cpu().numpy()，
     否则 np.argmax 会抛 "can't convert cuda:0 device type tensor to numpy"。
  2. masks.xy 是相对输入图的像素坐标，返回前要按原图 W/H 归一化。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# 注意导入顺序：cv2 必须先于 torch，否则会触发
# "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized"
# （conda 的 MKL 和 CUDA 版 torch 各带一份 libiomp5md.dll）
from ultralytics import YOLO

from .. import config

MAX_VERTICES = 64          # 与契约 §2.6 的 max_vertices 默认值保持一致
SIMPLIFY_EPS_RATIO = 0.004  # approxPolyDP 的 eps 占周长比例，与 configs/default.yaml 一致


def _simplify(pts: np.ndarray, max_vertices: int = MAX_VERTICES) -> np.ndarray:
    """把 mask 轮廓简化到 max_vertices 以内。

    整图预测通常只有几十个点，但放大 ROI 后能到几百上千个 —— 那种多边形人根本没法拖。
    """
    contour = pts.reshape(-1, 1, 2).astype(np.float32)
    peri = cv2.arcLength(contour, True)
    if peri <= 0:
        return pts

    eps = max(0.5, SIMPLIFY_EPS_RATIO * peri)
    approx = cv2.approxPolyDP(contour, eps, True)
    # 顶点仍然过多就逐步加大 eps
    guard = 0
    while len(approx) > max_vertices and guard < 12:
        eps *= 1.6
        approx = cv2.approxPolyDP(contour, eps, True)
        guard += 1

    out = approx.reshape(-1, 2)
    return out if len(out) >= 3 else pts


class Predictor:
    def __init__(self) -> None:
        if not config.WEIGHTS.is_file():
            raise RuntimeError(f"权重不存在: {config.WEIGHTS}")
        t0 = time.perf_counter()
        self.model = YOLO(str(config.WEIGHTS))
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.device = config.resolve_device()
        self.names = self.model.names
        self.model_version = self._make_version()

        # 预热一次，把 CUDA context / cuDNN autotune 的一次性开销挡在服务启动阶段
        self._warmup()

    def _make_version(self) -> str:
        import hashlib
        h = hashlib.sha256()
        with open(config.WEIGHTS, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return f"{config.WEIGHTS.name}@sha256:{h.hexdigest()[:8]}"

    def _warmup(self) -> None:
        """预热。

        别用全零图 —— 零图通常检不出东西，NMS、掩码轮廓提取这些最贵的后处理会被整段跳过，
        预热等于没做。实测用全零预热后，第一个真实请求仍要 ~100ms。
        这里改用随机噪声，让检测分支真正跑一遍。
        """
        rng = np.random.default_rng(0)
        for shape in ((640, 640, 3), (480, 640, 3)):
            try:
                dummy = rng.integers(0, 256, size=shape, dtype=np.uint8)
                self.model.predict(dummy, imgsz=config.IMGSZ, conf=0.05,
                                   verbose=False, device=self.device)
            except Exception:
                # 预热失败不该拦服务启动，真请求会再报错
                pass

    # ------------------------------------------------------------ 推理

    def predict_image(self, image_path: Path) -> tuple[list[dict], float, tuple[int, int]]:
        """返回 (instances, latency_ms, (W, H))。instances 的 polygon/bbox 已归一化。"""
        t0 = time.perf_counter()
        result = self.model.predict(
            str(image_path),
            imgsz=config.IMGSZ,
            conf=config.CONF,
            verbose=False,
            device=self.device,
        )[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        h, w = result.orig_shape[0], result.orig_shape[1]
        instances: list[dict] = []

        if result.masks is not None and result.boxes is not None:
            # CUDA 上这两者是 tensor，先搬回 CPU
            polys = result.masks.xy                      # list[np.ndarray]，已是 numpy
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)

            for pts, conf, cls_id in zip(polys, confs, clss):
                if pts is None or len(pts) < 3:
                    continue
                pts = _simplify(np.asarray(pts, dtype=np.float32))
                if len(pts) < 3:
                    continue

                poly = [[round(float(x) / w, 5), round(float(y) / h, 5)] for x, y in pts]
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                instances.append({
                    "class_id": int(cls_id),
                    "class_name": str(self.names.get(int(cls_id), "")),
                    "score": round(float(conf), 4),
                    "polygon": poly,
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                })

        instances.sort(key=lambda d: -d["score"])
        return instances, latency_ms, (int(w), int(h))

    # ------------------------------------------------------------ 缓存

    def cache_pred(self, image_id: str, payload: dict) -> Path | None:
        """把本次预标注结果落盘。供 /api/annotations 提交时取 pred_snapshot 用。

        契约 §4.2：必须落盘而不是放内存 —— 进程重启不丢，出问题能直接打开看。
        但**写缓存失败绝不能让预测请求失败**：标注结果是主数据，pred_snapshot 只是旁证。
        所以这里 best-effort，异常只打日志。
        """
        try:
            config.PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            p = config.PRED_CACHE_DIR / f"{_safe(image_id)}.json"
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(p)
            return p
        except Exception as e:      # noqa: BLE001 —— 这里就是要吞掉所有异常
            print(f"[warn] pred 缓存写入失败（不影响本次预测）: {image_id} -> {e}")
            return None

    def read_cached_pred(self, image_id: str) -> dict | None:
        p = config.PRED_CACHE_DIR / f"{_safe(image_id)}.json"
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def describe(self) -> dict:
        return {
            "weights": config.WEIGHTS.name,
            "model_version": self.model_version,
            "device": self.device,
            "imgsz": config.IMGSZ,
            "conf": config.CONF,
            "nc": len(self.names),
            "load_ms": round(self.load_ms, 1),
        }


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def utcnow() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
