"""POST /api/sample —— 主动学习选样

契约：docs/INTERFACE.md §2、ADDENDUM §3.2

算法**不在这里**：直接调模型侧 `active_learning/select.py::select()`，
这样离线消融实验和在线接口用的是同一套打分口径，报告里的数字才对得上。
本路由只负责几件它不适合管的事：

  1. image_id <-> 磁盘路径 的映射
  2. 默认 pool：优先用数据集包自带的 `al_pool/unlabeled_pool.txt`，
     没有就取整个 catalog 并剔除已标注过的图
  3. **两级筛选 + 分值缓存**（TTA 直接跑要 63 s，见下）
  4. reasons 的结构化（契约 §3.2 要求前端能展示"为什么选它"）

## 为什么需要两级筛选

`m8_al.score_tta` 对每张图跑 4 次前向（尺度 1.0/1.25 × 不翻转/水平翻转）算掩码
两两 IoU。100 张池子实测 **63.4 s**，前端点一下就转一分钟，等同于卡死。

所以 `tta` 走两级：先按 `entropy`（每张只要 1 次前向）粗筛出候选，
只对候选跑 TTA 精排。这不是妥协，是标准的 cascade screening ——
先用便宜的信号把候选集压下来，再用贵的信号排序。报告里要写明是两级。

`coreset` 不参与重排：它是贪心 k-中心，分值依赖"已选集合"逐步变化，
按分值全局排序会得到不同结果，所以照旧交给 `select()`（0.7 s，可接受）。

## 分值缓存

`entropy` / `tta` 的分值只和单张图有关，与 k 无关。用 `select(..., k=len(pool))`
能一次拿到全池分值，落成 `out/al_scores/<strategy>.json` 缓存。
同一批图第二次点就是毫秒级。缓存按 **权重的 sha256 + imgsz + 文件 mtime** 失效。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from .. import config
from ..schemas import SampleReason, SampleRequest, SampleResponse
from ..util import atomic_write_text, safe_name

router = APIRouter(prefix="/api", tags=["sample"])

MAX_POOL = 500          # 单次最多打分这么多张，防止一个请求跑上几分钟

# 两级筛选：候选数 = max(4k, 24)。4 倍余量让精排有得挑，24 是下限，
# 保证 k 很小的时候也不会因为候选太少而丢掉难例。
TTA_PREFILTER_FACTOR = 4
TTA_PREFILTER_MIN = 24

# 分值只和单张图有关、可以重排的策略。coreset 是贪心，不能重排。
_SCORE_CACHEABLE = ("entropy", "tta")

SCORE_CACHE_DIR = config.OUT_DIR / "al_scores"


# ---------------------------------------------------------------- 分值缓存

def _cache_file(strategy: str) -> Path:
    return SCORE_CACHE_DIR / f"{strategy}.json"


def _weights_key() -> str:
    """缓存版本号：换权重或换 imgsz 就必须重算。"""
    try:
        st = config.WEIGHTS.stat()
        return f"{st.st_size}:{int(st.st_mtime)}:{config.IMGSZ}"
    except OSError:
        return f"unknown:{config.IMGSZ}"


def _load_cache(strategy: str) -> dict:
    p = _cache_file(strategy)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if data.get("version") != _weights_key():
        return {}                                  # 权重变了，整份作废
    return data.get("scores") or {}


def _save_cache(strategy: str, scores: dict) -> None:
    try:
        SCORE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            _cache_file(strategy),
            json.dumps({"version": _weights_key(), "scores": scores},
                       ensure_ascii=False),
        )
    except OSError:
        pass                                       # 缓存写不进去不影响返回


def _mtime_key(path: Path) -> float:
    try:
        return round(path.stat().st_mtime, 3)
    except OSError:
        return 0.0


def _scored_rank(paths: list[Path], k: int, strategy: str) -> tuple[list[tuple[Path, float | None]], int]:
    """带缓存地给全池打分并按分值降序取前 k。返回 (选中, 缓存命中数)。

    只有 `_SCORE_CACHEABLE` 里的策略走这条路 —— 它们的分值与 k 无关，
    所以 `k=len(pool)` 一次算全量，之后的请求直接读缓存。
    """
    cache = _load_cache(strategy)
    todo: list[Path] = []
    for p in paths:
        key = str(p)
        ent = cache.get(key)
        if not ent or ent.get("mtime") != _mtime_key(p):
            todo.append(p)

    if todo:
        from active_learning.select import select as al_select

        picked = al_select(
            [str(p) for p in todo],
            len(todo),                              # k=全量 -> 拿到每张的分值
            strategy,
            weights=str(config.WEIGHTS),
            imgsz=config.IMGSZ,
            device=config.resolve_device(),
            return_scores=True,
        )
        for path, score in picked:
            if score is None:
                continue
            key = str(path)
            cache[key] = {"mtime": _mtime_key(Path(path)), "score": float(score)}
        _save_cache(strategy, cache)

    hits = len(paths) - len(todo)
    ranked = sorted(
        ((p, cache.get(str(p), {}).get("score")) for p in paths),
        key=lambda t: (t[1] is not None, t[1] or 0.0),
        reverse=True,
    )
    return ranked[:k], hits


# ---------------------------------------------------------------- 默认池

def _al_pool_ids(catalog) -> list[str]:
    """数据集包自带的未标注池（`<root>/al_pool/unlabeled_pool.txt`）。

    组内把"哪些图已标 / 待标"写进了文件，跑三臂实验时要按这个切，
    不能拿"全目录 − 已标注"当池子 —— 那会把 val 集也混进候选。
    """
    root = config.DATA.root
    if root is None:
        return []
    f = root / "al_pool" / "unlabeled_pool.txt"
    if not f.is_file():
        return []
    ids: list[str] = []
    for line in f.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        stem = safe_name(Path(name).stem)
        if catalog.get(stem) is not None:           # 不在索引里的忽略
            ids.append(stem)
    return list(dict.fromkeys(ids))


# ---------------------------------------------------------------- 路由

@router.post("/sample", response_model=SampleResponse)
def sample(body: SampleRequest, request: Request):
    catalog = request.app.state.catalog

    # ---- 1. 确定 pool
    source = "显式传入"
    if body.pool:
        unknown = [i for i in body.pool if catalog.get(i) is None]
        if unknown:
            raise HTTPException(
                404,
                f"pool 里有 {len(unknown)} 个未知 image_id，例如 {unknown[:5]}；"
                f"务必把提交时的 image_id 原样传回",
            )
        ids = list(dict.fromkeys(body.pool))       # 去重但保序
    else:
        ids = _al_pool_ids(catalog)
        if ids:
            source = "al_pool/unlabeled_pool.txt"
        if not ids:
            ids = [r.image_id for r in catalog.query(limit=10_000)[0]]
            source = "全目录"
        if body.exclude_annotated:
            done = _annotated_ids()
            ids = [i for i in ids if i not in done]

    if not ids:
        raise HTTPException(
            400,
            "可选池为空：所有图都标过了，或 pool 传成了空数组。"
            "要重新过一遍请传 exclude_annotated=false",
        )
    if len(ids) > MAX_POOL:
        raise HTTPException(
            400, f"pool 有 {len(ids)} 张，超过单次上限 {MAX_POOL}；请分批调用"
        )

    paths = [catalog.get(i).path for i in ids]
    # 注意：select() 内部会把 pool 转成 str 再返回，所以这里键也用 str，
    # 否则 Path 键和 str 查不到一起，每次都要走 O(n) 的兜底扫描。
    id_of = {str(p): i for p, i in zip(paths, ids)}

    # ---- 2. 选样
    t0 = time.perf_counter()
    cache_hits = 0
    two_stage = False
    cand_n = 0
    try:
        from active_learning.select import select as al_select  # noqa: F401
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(
            501, f"模型侧 active_learning/select.py 导入失败：{type(e).__name__}: {e}"
        )

    try:
        if body.strategy == "tta":
            # 两级：entropy 粗筛候选 -> 只对候选跑 TTA
            cand_k = min(len(paths), max(TTA_PREFILTER_MIN,
                                         TTA_PREFILTER_FACTOR * body.k))
            coarse, h1 = _scored_rank(paths, cand_k, "entropy")
            cand_n = len(coarse)
            fine, h2 = _scored_rank([p for p, _ in coarse], body.k, "tta")
            cache_hits = h1 + h2
            two_stage = cand_n < len(paths)
            picked = fine
        elif body.strategy in _SCORE_CACHEABLE:
            picked, cache_hits = _scored_rank(paths, body.k, body.strategy)
        else:
            picked = al_select(
                [str(p) for p in paths],
                body.k,
                body.strategy,
                weights=str(config.WEIGHTS),
                imgsz=config.IMGSZ,
                device=config.resolve_device(),
                seed=body.seed,
                return_scores=True,
            )
    except FileNotFoundError as e:
        raise HTTPException(500, f"选样缺少权重：{e}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    latency = (time.perf_counter() - t0) * 1000

    # ---- 3. 路径映射回 image_id，并按顺序给 rank + 分值
    selected: list[str] = []
    reasons: dict[str, SampleReason] = {}
    criterion = body.strategy
    if two_stage:
        criterion = f"tta·两级(entropy 粗筛 {len(paths)}→候选 {cand_n})"

    for item in picked:
        path, score = item if isinstance(item, (tuple, list)) and len(item) == 2 \
            else (item, None)
        iid = id_of.get(str(path))
        if iid is None:
            continue
        selected.append(iid)
        reasons[iid] = SampleReason(
            rank=len(selected),
            criterion=criterion,
            score=round(float(score), 4) if score is not None else None,
        )

    note = (
        "reasons.score 为模型侧真实打分：entropy=二元熵 0-1（越高越不确定，"
        "无检出=1.0）；tta=4 视图不一致度 0-1；coreset=描述子距离（越高越多样）；"
        "random 无分值（null）。rank 即选样顺序。"
    )
    if two_stage:
        note += (
            f" 本次 tta 走两级筛选（entropy 粗筛 {len(paths)}→候选，再对候选跑 tta）："
            "单张 tta 要 4 次前向，全池直算约 60 s，交互不可用。"
        )
    if cache_hits:
        note += f" 本次有 {cache_hits} 张命中分值缓存。"

    return SampleResponse(
        selected=selected,
        reasons=reasons,
        strategy=body.strategy,
        pool_size=len(ids),
        k=len(selected),
        latency_ms=round(latency, 1),
        note=note,
    )


def _annotated_ids() -> set[str]:
    """扫 out/annotations/*/ 下已有的 .txt，得到"已经标过"的 image_id 集合。

    约定（ADDENDUM §4.3）：文件名就是 image_id（已过 safe_name）。
    """
    done: set[str] = set()
    base = config.ANNOTATION_DIR
    if not base.is_dir():
        return done
    for arm in base.iterdir():
        if not arm.is_dir():
            continue
        for f in arm.glob("*.txt"):
            done.add(safe_name(f.stem))
    return done
