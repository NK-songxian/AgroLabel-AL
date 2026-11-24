"""`backend/tools/human_stats.py` 的自测。

只测统计逻辑，不碰真实 out/ —— 全部在 tmp_path 里造数据。
运行：

    .venv\\Scripts\\python.exe -m pytest backend/tests/test_human_stats.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "backend" / "tools"))

import human_stats as hs  # noqa: E402


def _write(root: Path, arm: str, image_id: str, elapsed_ms: int, operator: str,
           ts: str, elapsed_from_open: int | None = None,
           label_state: str = "labeled", n_instances: int = 1) -> None:
    d = root / arm
    d.mkdir(parents=True, exist_ok=True)
    rec = {
        "image_id": image_id,
        "label_state": label_state,
        "arm": arm,
        "n_instances": n_instances,
        "pred_cached": arm != "manual",
        "model_version": "test@sha256:00000000",
        "elapsed_ms": elapsed_ms,
        "elapsed_from_open_ms": elapsed_from_open if elapsed_from_open is not None else elapsed_ms,
        "operator": operator,
        "session_id": "t",
        "ts": ts,
    }
    (d / f"{image_id}.json").write_text(
        json.dumps(rec, ensure_ascii=False), encoding="utf-8"
    )


def _ts(i: int) -> str:
    """第 i 张的时间戳，保证升序。"""
    return f"2026-11-20T10:{i:02d}:00+08:00"


def test_median_and_persons(tmp_path: Path):
    """三臂各 5 张、两个标注员：中位按人也能算出来，operator 为空单列。"""
    for arm in ("manual", "assisted"):
        for i, ms in enumerate((30_000, 40_000, 50_000, 60_000, 70_000)):
            who = "张三" if i < 3 else "李四"
            _write(tmp_path, arm, f"img_{i}", ms, who, _ts(i))

    data = hs.load_sidecars(tmp_path, ("manual", "assisted"))
    cleaned = hs.clean_arm(data["manual"], cap_s=900, drop_warmup=0, mad_k=0)
    row = hs.arm_summary("manual", cleaned, ops={}, edit=None)

    assert row["n_images"] == 5
    assert row["n_persons"] == 2
    assert row["median_ms"] == pytest.approx(50_000, rel=1e-6)
    assert row["p90_ms"] == pytest.approx(66_000, rel=1e-6)   # 4/5 处线性插值
    assert row["persons"]["张三"]["n"] == 3
    assert row["persons"]["李四"]["n"] == 2
    assert row["unnamed"] is False


def test_all_blank_operator_marks_unnamed(tmp_path: Path):
    """operator 全空 -> 打 unnamed 标记（不允许当真人数据引用）。"""
    for i in range(3):
        _write(tmp_path, "manual", f"img_{i}", 30_000, "", _ts(i))

    data = hs.load_sidecars(tmp_path, ("manual",))
    cleaned = hs.clean_arm(data["manual"], cap_s=900, drop_warmup=0, mad_k=0)
    row = hs.arm_summary("manual", cleaned, ops={}, edit=None)

    assert row["unnamed"] is True
    assert row["n_persons"] == 0
    assert hs.UNKNOWN_OPERATOR in row["persons"]


def test_cap_and_mad_exclusion(tmp_path: Path):
    """超上限的离开记录 + 一个离群值都要被剔，并且计数可核对。"""
    # 10 张正常 40 s，1 张 2 小时（离开），1 张 200 s（离群）
    normal = [40_000] * 10
    for i, ms in enumerate(normal):
        _write(tmp_path, "manual", f"n_{i}", ms, "张三", _ts(i))
    _write(tmp_path, "manual", "away", 7_200_000, "张三", _ts(10))    # 2 h
    _write(tmp_path, "manual", "spike", 200_000, "张三", _ts(11))     # 5x

    data = hs.load_sidecars(tmp_path, ("manual",))
    cleaned = hs.clean_arm(data["manual"], cap_s=900, drop_warmup=0, mad_k=3.0)

    assert cleaned["over_cap"] == 1          # 2 h 那条被上限拦掉
    # 正常值全等 -> MAD=0 -> 离群规则自动跳过（这是刻意的保守行为）
    assert cleaned["mad_outlier"] == 0
    assert cleaned["n_kept"] == 11

    # 换成有波动的正常值，离群规则就会生效
    tmp2 = tmp_path / "b"
    for i, ms in enumerate((35_000, 38_000, 40_000, 42_000, 45_000, 40_000)):
        _write(tmp2, "manual", f"m_{i}", ms, "张三", _ts(i))
    _write(tmp2, "manual", "spike", 300_000, "张三", _ts(6))
    data2 = hs.load_sidecars(tmp2, ("manual",))
    cleaned2 = hs.clean_arm(data2["manual"], cap_s=900, drop_warmup=0, mad_k=3.0)
    assert cleaned2["mad_outlier"] == 1
    assert cleaned2["n_kept"] == 6


def test_warmup_skipped_when_not_enough_data(tmp_path: Path):
    """数据量 <= 预热张数时不许把整臂剔空。"""
    for i in range(2):
        _write(tmp_path, "manual", f"img_{i}", 30_000, "张三", _ts(i))

    data = hs.load_sidecars(tmp_path, ("manual",))
    cleaned = hs.clean_arm(data["manual"], cap_s=900, drop_warmup=3, mad_k=0)

    assert cleaned["warmup"] == 0
    assert cleaned["warmup_skipped"] is True
    assert cleaned["n_kept"] == 2

    # 数据够多时才剔
    tmp2 = tmp_path / "many"
    for i in range(6):
        _write(tmp2, "manual", f"img_{i}", 30_000, "张三", _ts(i))
    data2 = hs.load_sidecars(tmp2, ("manual",))
    cleaned2 = hs.clean_arm(data2["manual"], cap_s=900, drop_warmup=3, mad_k=0)
    assert cleaned2["warmup"] == 3
    assert cleaned2["n_kept"] == 3


def test_op_counts_only_count_kept_images(tmp_path: Path):
    """ops 计数必须按 sidecar 里的图过滤，不能把没参与统计的图算进来。"""
    for i in range(2):
        _write(tmp_path, "assisted", f"img_{i}", 30_000, "张三", _ts(i))

    ops_dir = tmp_path / "logs"
    ops_dir.mkdir()
    (ops_dir / "ops_assisted.csv").write_text(
        "ts,image,op,instance,detail,n_ops_on_image,api_ms\n"
        "2026-11-20 10:00:01,img_0,删实例,2,Tomato,1,\n"
        "2026-11-20 10:00:02,img_0,改类别,3,Apple -> Tomato,2,\n"
        "2026-11-20 10:00:03,img_9,删实例,1,别的图,1,\n",       # img_9 不在 sidecar 里
        encoding="utf-8-sig",
    )

    data = hs.load_sidecars(tmp_path, ("assisted",))
    cleaned = hs.clean_arm(data["assisted"], cap_s=900, drop_warmup=0, mad_k=0)
    keep = {r["image_id"] for r in cleaned["kept"]}
    by_op = hs.load_op_counts(ops_dir / "ops_assisted.csv", keep)
    total = hs.sum_ops(by_op)

    assert total["删实例"] == 1
    assert total["改类别"] == 1
    assert sum(total.values()) == 2          # img_9 的没被算进来


def test_markdown_renders_without_editcost(tmp_path: Path):
    """没有 m6 汇总时，编辑代价列应显示 — 而不是崩或写 None。"""
    for i in range(4):
        _write(tmp_path, "manual", f"img_{i}", 30_000, "张三", _ts(i))
    data = hs.load_sidecars(tmp_path, ("manual",))
    cleaned = hs.clean_arm(data["manual"], cap_s=900, drop_warmup=0, mad_k=0)
    rows = [hs.arm_summary("manual", cleaned, ops={}, edit=None)]

    class A:
        timing = "both"

    md = hs.render_md(rows, A(), [])
    assert "纯人工" in md
    assert "—" in md
    assert "None" not in md
