#!/usr/bin/env python
"""AgroLabel-AL 标注服务入口。

启动（在仓库根目录）：
    .\\.venv\\Scripts\\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload

Swagger: http://127.0.0.1:8000/docs

本文件只做三件事：环境兜底、生命周期里加载一次模型、挂路由。
"""

from __future__ import annotations

# ⚠️ 必须在 import torch 之前完成这几步（实测会崩）：
#   1) KMP_DUPLICATE_LIB_OK —— conda 的 MKL 与 CUDA 版 torch 各带一份 libiomp5md.dll
#   2) import cv2 早于 torch —— 否则 OMP Error #15
#   3) YOLO_OFFLINE    —— ultralytics 有联网自检（字体/AMP/权重探测），
#      断网或挂代理时会卡住几十秒。团队已在 training/train_seg_baseline.py
#      统一默认开启（commit 5c2042c），后端跟着对齐，别让推理路径也去联网。
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("YOLO_OFFLINE", "1")

import cv2  # noqa: E402,F401  （顺序敏感，勿删）

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from . import config  # noqa: E402
from .model_bridge.catalog import Catalog  # noqa: E402
from .model_bridge.predictor import Predictor  # noqa: E402
from .routers import (  # noqa: E402
    annotations,
    editcost,
    images,
    logs,
    predict,
    refine,
    sample,
    system,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动阶段做两件重活，都只做一次
    print(f"[startup] 数据源: {config.DATA.mode} —— {config.DATA.note}")
    catalog = Catalog()
    catalog.build()
    print(f"[startup] 索引 {len(catalog)} 张，跳过 {catalog.skipped} 条（文件不存在）")

    print(f"[startup] 加载权重: {config.WEIGHTS.name}")
    predictor = Predictor()
    print(f"[startup] 模型就绪 device={predictor.device} "
          f"nc={len(predictor.names)} load={predictor.load_ms:.0f}ms")

    app.state.catalog = catalog
    app.state.predictor = predictor
    yield
    # 无需显式释放：进程退出即回收显存


app = FastAPI(
    title="AgroLabel-AL 标注服务",
    version="0.1.0",
    description="赛题一 · 预标注与主动学习原型系统的后端。契约见 docs/INTERFACE.md 与 docs/INTERFACE_ADDENDUM.md",
    lifespan=lifespan,
)

# Vite 开发服务器默认 5173；独立预览用 4173
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173", "http://localhost:5173",
        "http://127.0.0.1:4173", "http://localhost:4173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(images.router)
app.include_router(predict.router)
app.include_router(annotations.router)
app.include_router(logs.router)
app.include_router(sample.router)
app.include_router(refine.router)
app.include_router(editcost.router)


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "AgroLabel-AL backend",
        "docs": "/docs",
        "health": "/api/health",
    }
