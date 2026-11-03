"""主动学习选样入口：`select(pool, k, strategy) -> list[str]`。

对应 `docs/HANDOFF_FRONTEND_BACKEND.md` §6 约定的签名，供后端 `/api/sample` 调用。

策略实现**直接复用 `src/m8_al.py` 的 `score_*`**，不另写一套 —— 保证
「离线消融实验」和「在线接口」用的是同一套打分口径，报告里的数字才对得上。

    random   随机抽（对照基线）
    entropy  置信度二元熵，越接近 0.5 越可疑
    tta      4 视图扰动一致性，越低越可疑
    coreset  颜色纹理描述子上的 k 中心贪心，纯多样性

用法（自测）::

    python active_learning/select.py --pool "F:/_plantseg/dataset_plantseg_34/images/val/*.jpg" \
        --k 5 --strategy random
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import m8_al as _al  # noqa: E402  （模块级无副作用，只有函数定义）

STRATEGIES = ("random", "entropy", "tta", "coreset")


class _ScoreArgs:
    """m8 的 score_* 只读这三个属性。"""

    def __init__(self, imgsz: int, device: str):
        self.imgsz = imgsz
        self.device = device
        self._desc_cache: dict = {}


def _default_weights() -> Path:
    cands = sorted((PROJECT / "weights").glob("baseline_*.pt"))
    if cands:
        return cands[-1]
    runs = sorted((PROJECT / "runs").rglob("best.pt"))
    if runs:
        return runs[-1]
    raise FileNotFoundError(
        "找不到权重：请放 weights/baseline_*.pt，或传 weights= 指定 .pt")


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def select(
    pool,
    k: int,
    strategy: str = "random",
    *,
    weights=None,
    imgsz: int = 640,
    device=None,
    seed: int = 0,
    labeled=None,
    return_scores: bool = False,
):
    """从 `pool`（图片路径列表）里选 `k` 张。

    返回
    ----
    return_scores=False（默认，兼容旧签名）: 被选中的路径列表 list[str]
    return_scores=True : list[tuple[str, float | None]]，(路径, 分值) 按选样顺序。
        分值含义随策略：entropy=二元熵 0-1（越高越不确定）、tta=不一致度 0-1、
        coreset=离已选中心的描述子距离（越高越多样）、random=None（无意义）。

    参数
    ----
    strategy : random | entropy | tta | coreset
    labeled  : 已标注图路径（仅 coreset 用，让第一次选择偏向「和已有数据最不像」）
    """
    pool = [str(p) for p in pool]
    k = max(0, min(int(k), len(pool)))
    if k == 0:
        return []

    strategy = (strategy or "random").lower()
    if strategy not in STRATEGIES:
        raise ValueError(f"未知策略 {strategy!r}，可选 {STRATEGIES}")

    if strategy == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pool), size=k, replace=False)
        if return_scores:
            return [(pool[i], None) for i in idx]
        return [pool[i] for i in idx]

    # 其余策略需要模型
    from ultralytics import YOLO

    wpath = Path(weights) if weights else _default_weights()
    dev = str(device) if device not in (None, "") else ("0" if _cuda_available() else "cpu")
    model = YOLO(str(wpath))
    nc = len(model.names)
    args = _ScoreArgs(imgsz, dev)

    if strategy == "entropy":
        idx, sc = _al.score_entropy(pool, model, nc, k, args)
    elif strategy == "tta":
        idx, sc = _al.score_tta(pool, model, nc, k, args)
    else:  # coreset
        idx, sc = _al.score_coreset(pool, model, nc, k, args, labeled=labeled)
    if return_scores:
        return [(pool[i], (float(s) if s is not None else None))
                for i, s in zip(idx, sc)]
    return [pool[i] for i in idx]


def _main() -> None:
    import argparse
    import glob

    ap = argparse.ArgumentParser(description="主动学习选样入口自测")
    ap.add_argument("--pool", required=True, help="glob 或目录")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--strategy", default="random", choices=STRATEGIES)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    p = Path(args.pool)
    if p.is_dir():
        pool = sorted(str(x) for x in p.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png"})
    else:
        pool = sorted(glob.glob(args.pool))
    print(f"pool={len(pool)}  strategy={args.strategy}  k={args.k}")
    picked = select(pool, args.k, args.strategy, weights=args.weights,
                    imgsz=args.imgsz, device=args.device, return_scores=True)
    for x, sc in picked:
        print(f"   {Path(x).name}  score={sc if sc is None else round(sc, 4)}")


if __name__ == "__main__":
    _main()
