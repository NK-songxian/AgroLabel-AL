"""三臂人工提效统计：把 out/annotations/<arm>/*.json 汇总成手册 §08 那张表。

## 它解决什么

手册 §08 要求三臂对照（纯人工 / 预标注+修正 / 预标注·禁用点提示），
列包括「图数 / 人数 / 中位完成时间 / 漏画·删除·修边 / 最终质检质量」。

每张图的真实耗时和操作员其实**已经在标注时落盘**了
（`out/annotations/<arm>/<image_id>.json` 的 `elapsed_ms` / `operator`），
但没人读。这个脚本只做一件事：读出来、按臂按人汇总、按规则剔除异常、出表。

## 口径（提案 docs/PROPOSAL_human_stats.md 的推荐值，均可用 CLI 覆盖）

计时有两个口径，两个都算：

    elapsed_ms            图上**第一个标注动作** → 落盘   ← 主口径（不含看图）
    elapsed_from_open_ms  打开这张图 → 落盘             ← 对照（含看图/熟悉）

剔除规则（默认全开）：

    1. 预热   每臂按 ts 升序剔掉前 --drop-warmup 张（默认 3）—— 手册要求"记录预热规则"
    2. 上限   --exclude-above 秒（默认 900 = 15 min）以上视为离开，剔除
    3. 离群   每臂按 MAD 剔 --mad-k 倍中位绝对偏差以外（默认 3；MAD=0 时跳过）

**剔除必须可见**：报告里逐臂列出各项剔了几张，并同时给"剔除前/后"中位数。
两个数差得远，说明数据本身有问题，不要只贴好看的那个。

## 诚实性

脚本只汇总**真人**标注。若某臂的 operator 全为空，该臂会被标成
「⚠ 未署名/模拟口径」，不允许与有名有姓的臂混在一起当真人数据引用。

## 用法

    python backend/tools/human_stats.py
    python backend/tools/human_stats.py --timing both --drop-warmup 3 --mad-k 3
    python backend/tools/human_stats.py --with-editcost --json out/report/human_stats.json

只读 out/ 下的文件，**不写任何标注数据**，不新增依赖（纯标准库）。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]

ARM_ORDER = ("manual", "assisted", "noprompt")
ARM_LABEL = {
    "manual": "纯人工",
    "assisted": "预标注 + 修正",
    "noprompt": "预标注 · 禁用点提示",
}

UNKNOWN_OPERATOR = "（未填写）"


# ---------------------------------------------------------------- 读数据

def load_sidecars(base: Path, arms: tuple[str, ...]) -> dict[str, list[dict]]:
    """读每个臂的 sidecar。缺目录 / 缺文件都当"这个臂还没标"，不算错。"""
    out: dict[str, list[dict]] = {}
    for arm in arms:
        d = base / arm
        rows: list[dict] = []
        if d.is_dir():
            for p in sorted(d.glob("*.json")):
                try:
                    rec = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                # 只认 sidecar：必须有 elapsed_ms 或 ts，避免把别的 json 混进来
                if "elapsed_ms" not in rec and "ts" not in rec:
                    continue
                rec["_file"] = p.name
                rows.append(rec)
        out[arm] = rows
    return out


def _ts_key(rec: dict) -> datetime:
    """按 ts 排序用。解析不了就排最前（当作最早的），不要让一条坏数据打断统计。"""
    raw = str(rec.get("ts") or "")
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return datetime.min


# ---------------------------------------------------------------- 基础统计

def _median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 1) if xs else None


def _pct(xs: list[float], q: float) -> float | None:
    """线性插值分位数（p 取 0~1）。样本少时也能给个数，不做外推。"""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return round(s[0], 1)
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return round(s[lo] + (s[hi] - s[lo]) * frac, 1)


def _mad_exclude(values: list[float], k: float) -> set[int]:
    """返回应剔除的下标集合（中位绝对偏差法）。MAD=0 时返回空集并置标记。"""
    if k <= 0 or len(values) < 5:
        return set()                      # 样本太少时离群判定没有意义
    med = statistics.median(values)
    devs = [abs(v - med) for v in values]
    mad = statistics.median(devs)
    if mad == 0:
        return set()                      # 一半以上完全相同 -> 判不了，跳过
    limit = k * 1.4826 * mad              # 1.4826：正态下 MAD -> 标准差的换算
    return {i for i, v in enumerate(values) if abs(v - med) > limit}


def clean_arm(recs: list[dict], cap_s: float, drop_warmup: int, mad_k: float) -> dict:
    """对一个臂做剔除，返回保留记录 + 逐项剔除计数。

    顺序有意为之：**先预热、再上限、最后离群**。
    预热剔的是"学习曲线"，上限剔的是"人离开了"，离群剔的是"统计尾巴"。
    顺序换了数会对不上，所以要写死。
    """
    ordered = sorted(recs, key=_ts_key)

    # 预热只在"确实还有剩余数据"时才剔。否则一次 pilot（1~2 张）会被整臂剔空，
    # 报告变成"没有可用记录"，比不剔还误导。
    warmup_n = drop_warmup if (drop_warmup > 0 and len(ordered) > drop_warmup) else 0
    warmup_dropped = ordered[:warmup_n]
    rest = ordered[warmup_n:]

    kept: list[dict] = []
    over_cap = 0
    for r in rest:
        ms = r.get("elapsed_ms")
        if not isinstance(ms, (int, float)):
            continue                      # 没有耗时的记录不参与时间统计
        if cap_s > 0 and ms > cap_s * 1000:
            over_cap += 1
            continue
        kept.append(r)

    # 离群在"已过上限"的集合上算，避免一个 2 小时的 outlier 把 MAD 抬得没法用
    vals = [float(r["elapsed_ms"]) for r in kept]
    mad_idx = _mad_exclude(vals, mad_k)
    if mad_idx:
        kept = [r for i, r in enumerate(kept) if i not in mad_idx]

    return {
        "kept": kept,
        "n_raw": len(recs),
        "warmup": len(warmup_dropped),
        "warmup_skipped": drop_warmup > 0 and warmup_n == 0,
        "over_cap": over_cap,
        "mad_outlier": len(mad_idx),
        "no_timing": sum(1 for r in rest if not isinstance(r.get("elapsed_ms"), (int, float))),
        "n_kept": len(kept),
        "_vals_all": [float(r["elapsed_ms"]) for r in rest
                      if isinstance(r.get("elapsed_ms"), (int, float))],
    }


# ---------------------------------------------------------------- ops 补充

def load_op_counts(ops_csv: Path, keep_ids: set[str]) -> dict[str, Counter]:
    """从 out/logs/ops_<arm>.csv 数每张图的各类操作数。

    只统计 keep_ids 里的图 —— 分母必须和 sidecar 那批一致，否则
    "删实例 232 次"可能来自根本没参与统计的图。
    """
    by_op: dict[str, Counter] = defaultdict(Counter)
    if not ops_csv.is_file():
        return by_op
    try:
        with open(ops_csv, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                img = (row.get("image") or "").strip()
                if not img or (keep_ids and img not in keep_ids):
                    continue
                op = (row.get("op") or "").strip()
                if op:
                    by_op[img][op] += 1
    except OSError:
        return by_op
    return by_op


def sum_ops(by_op: dict[str, Counter]) -> Counter:
    total: Counter = Counter()
    for c in by_op.values():
        total.update(c)
    return total


# ---------------------------------------------------------------- m6 合并

def load_editcost(report_dir: Path, arm: str) -> dict | None:
    """读 m6_editcost.py 的汇总 json（不重算）。

    m6 的产物名由 --name 决定，常见写法是 <arm> 或 manual_xxx，
    这里按"精确名 > 前缀匹配"找，多个命中就取最新的那个文件。
    """
    if not report_dir.is_dir():
        return None
    cands = [p for p in report_dir.glob("*.json") if p.stem == arm]
    if not cands:
        cands = [p for p in report_dir.glob(f"{arm}*.json")]
    if not cands:
        return None
    p = max(cands, key=lambda x: x.stat().st_mtime)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    first = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else None)
    if not isinstance(first, dict):
        return None
    return {
        "source": p.name,
        "n_images": first.get("n_images") or first.get("n"),
        "weighted_per_image": first.get("weighted_per_image"),
        "weighted_vs_manual": first.get("weighted_vs_manual"),
        "matched_mask_iou": first.get("matched_mask_iou"),
        "instance_recall": first.get("instance_recall"),
    }


# ---------------------------------------------------------------- 汇总

def arm_summary(arm: str, cleaned: dict, ops: Counter, edit: dict | None) -> dict:
    kept = cleaned["kept"]
    # kept 里每条都保证有数值型 elapsed_ms（clean_arm 已过滤），这里直接取
    el = [float(r["elapsed_ms"]) for r in kept]
    el_open = [float(r["elapsed_from_open_ms"]) for r in kept
               if isinstance(r.get("elapsed_from_open_ms"), (int, float))]

    # 按人拆：operator 为空单列 unknown，不静默丢
    persons: dict[str, list[float]] = defaultdict(list)
    for r in kept:
        who = (r.get("operator") or "").strip() or UNKNOWN_OPERATOR
        persons[who].append(float(r["elapsed_ms"]))

    unnamed = set(persons) == {UNKNOWN_OPERATOR} and bool(kept)
    states = Counter((r.get("label_state") or "?").strip() for r in kept)

    return {
        "arm": arm,
        "label": ARM_LABEL.get(arm, arm),
        "n_images": len(kept),
        "n_persons": len([p for p in persons if p != UNKNOWN_OPERATOR]),
        "unnamed": unnamed,
        "median_ms": _median(el),
        "mean_ms": round(statistics.mean(el), 1) if el else None,
        "p25_ms": _pct(el, 0.25),
        "p75_ms": _pct(el, 0.75),
        "p90_ms": _pct(el, 0.90),
        "median_open_ms": _median(el_open),
        "persons": {k: {"n": len(v), "median_ms": _median(v)} for k, v in persons.items()},
        "label_state": dict(states),
        "n_instances": sum(int(r.get("n_instances") or 0) for r in kept),
        "ops": dict(ops),
        "edit": edit,
        "excluded": {
            "raw": cleaned["n_raw"],
            "warmup": cleaned["warmup"],
            "over_cap": cleaned["over_cap"],
            "mad_outlier": cleaned["mad_outlier"],
            "no_timing": cleaned["no_timing"],
        },
        "median_ms_before_filter": _median(cleaned["_vals_all"]),
    }


def _s(ms: float | None) -> str:
    return "—" if ms is None else f"{ms / 1000:.1f} s"


def _delta(new: float | None, old: float | None) -> str:
    """相对纯人工的降幅。旧值为 0 或缺失时不给数字，避免算出无穷大。"""
    if new is None or not old:
        return ""
    return f" ({(new - old) / old * 100:+.1f}%)"


def render_md(rows: list[dict], args, notes: list[str]) -> str:
    base = next((r for r in rows if r["arm"] == "manual" and r["median_ms"]), None)
    base_ms = base["median_ms"] if base else None

    L: list[str] = []
    L.append("## §08 人工提效（真人实测）")
    L.append("")
    L.append("| 臂 | 图数 | 人数 | 中位完成时间 | P90 | 加权编辑单位/张 | 掩码 IoU | 实例召回 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        name = r["label"] + (" ⚠ 未署名/模拟口径" if r["unnamed"] else "")
        if not r["n_images"]:
            L.append(f"| {name} | 0 | — | — | — | — | — | — |")
            continue

        e = r["edit"] or {}
        w = e.get("weighted_per_image")
        wv = e.get("weighted_vs_manual")
        wcell = "—" if w is None else f"{w:.2f}"
        if w is not None and wv is not None and r["arm"] != "manual":
            wcell += f" ({wv:+.1f}%)"

        iou = e.get("matched_mask_iou")
        rec = e.get("instance_recall")
        iou_cell = f"{iou:.3f}" if isinstance(iou, (int, float)) else "—"
        rec_cell = f"{rec:.3f}" if isinstance(rec, (int, float)) else "—"

        L.append(
            f"| {name} | {r['n_images']} | {r['n_persons']} | "
            f"{_s(r['median_ms'])}{_delta(r['median_ms'], base_ms)} | "
            f"{_s(r['p90_ms'])} | {wcell} | {iou_cell} | {rec_cell} |"
        )
    L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="三臂人工提效统计（读 out/annotations/<arm>/*.json）")
    ap.add_argument("--annotations", default=str(PROJECT / "out" / "annotations"))
    ap.add_argument("--ops", default=str(PROJECT / "out" / "logs"),
                    help="ops_<arm>.csv 所在目录；用于补 ops 计数")
    ap.add_argument("--report", default=str(PROJECT / "out" / "report"),
                    help="m6_editcost.py 的汇总 json 目录（--with-editcost 时读）")
    ap.add_argument("--arms", default=",".join(ARM_ORDER))
    ap.add_argument("--timing", choices=("first-action", "from-open", "both"),
                    default="both")
    ap.add_argument("--exclude-above", type=float, default=900.0,
                    help="单张超过多少秒视为离开并剔除（0 = 不设上限）")
    ap.add_argument("--drop-warmup", type=int, default=3,
                    help="每臂按 ts 升序剔掉前 N 张（0 = 不剔）")
    ap.add_argument("--mad-k", type=float, default=3.0,
                    help="MAD 离群倍数（0 = 不剔）")
    ap.add_argument("--with-editcost", action="store_true",
                    help="合并 m6_editcost.py 的汇总（加权编辑单位、IoU、召回）")
    ap.add_argument("--json", default=None, help="把结构化结果写到这个路径")
    ap.add_argument("--md", default=None, help="把 Markdown 表写到这个路径")
    args = ap.parse_args()

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    unknown_arms = [a for a in arms if a not in ARM_ORDER]
    if unknown_arms:
        raise SystemExit(f"未知实验臂: {unknown_arms}，可选 {list(ARM_ORDER)}")

    side = load_sidecars(Path(args.annotations), arms)

    rows: list[dict] = []
    for arm in arms:
        recs = side[arm]
        cleaned = clean_arm(recs, args.exclude_above, args.drop_warmup, args.mad_k)
        keep_ids = {str(r.get("image_id") or Path(str(r.get("_file", ""))).stem)
                    for r in cleaned["kept"]}
        by_op = load_op_counts(Path(args.ops) / f"ops_{arm}.csv", keep_ids)
        edit = load_editcost(Path(args.report), arm) if args.with_editcost else None
        rows.append(arm_summary(arm, cleaned, sum_ops(by_op), edit))

    # ---- 打印
    print()
    for r in rows:
        name = r["label"]
        if r["unnamed"]:
            name += "  ⚠ 未署名/模拟口径"
        print(f"[{r['arm']}] {name}")
        if not r["n_images"]:
            print(f"    没有可用记录（原始 {r['excluded']['raw']} 条；"
                  f"预热剔 {r['excluded']['warmup']}、超上限 {r['excluded']['over_cap']}、"
                  f"离群 {r['excluded']['mad_outlier']}）")
            print()
            continue
        print(f"    图数 {r['n_images']}   人数 {r['n_persons']}"
              + ("" if r["n_persons"] else "  ← 没有任何记录填了 operator"))
        print(f"    中位 {_s(r['median_ms'])}  P25 {_s(r['p25_ms'])}  "
              f"P75 {_s(r['p75_ms'])}  P90 {_s(r['p90_ms'])}  均值 {_s(r['mean_ms'])}")
        if args.timing in ("from-open", "both"):
            print(f"    对照口径（含看图）中位 {_s(r['median_open_ms'])}")
        if r["median_ms_before_filter"] and r["median_ms"]:
            print(f"    剔除前中位 {_s(r['median_ms_before_filter'])}"
                  f"（与剔除后 {_s(r['median_ms'])} 对比，差得远说明数据有问题）")
        print(f"    剔除：预热 {r['excluded']['warmup']} / 超上限 {r['excluded']['over_cap']}"
              f" / 离群 {r['excluded']['mad_outlier']}"
              f" / 无耗时 {r['excluded']['no_timing']}（原始 {r['excluded']['raw']} 条）")
        print(f"    状态 {r['label_state']}   实例合计 {r['n_instances']}")
        if r["ops"]:
            top = ", ".join(f"{k} {v}" for k, v in sorted(r["ops"].items(),
                                                          key=lambda kv: -kv[1])[:6])
            print(f"    ops：{top}")
        if r["edit"]:
            e = r["edit"]
            print(f"    m6（{e['source']}）：加权 {e['weighted_per_image']} 单位/张"
                  f"  IoU {e['matched_mask_iou']}  召回 {e['instance_recall']}")
        for who, p in sorted(r["persons"].items(), key=lambda kv: -kv[1]["n"]):
            print(f"      · {who}: {p['n']} 张，中位 {_s(p['median_ms'])}")
        print()

    # ---- 结论提示
    notes: list[str] = []
    total = sum(r["n_images"] for r in rows)
    if total == 0:
        notes.append("一条可用记录都没有 —— 先按三臂各标一遍再来。")
    if any(r["n_persons"] == 0 and r["n_images"] for r in rows):
        notes.append("有臂的记录全部没填 operator：按人统计失效，且不能当真人数据引用。")
    if args.timing == "both":
        notes.append("口径：中位完成时间 = 图上首个标注动作 → 落盘（elapsed_ms），"
                     "已按上述规则剔除；括号内为相对纯人工的降幅。")
    for n in notes:
        print(f"注：{n}")

    # ---- 落盘
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "params": {
            "annotations": args.annotations, "arms": list(arms),
            "timing": args.timing, "exclude_above_s": args.exclude_above,
            "drop_warmup": args.drop_warmup, "mad_k": args.mad_k,
            "with_editcost": args.with_editcost,
        },
        "arms": rows,
    }
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  JSON -> {p}")
    if args.md:
        p = Path(args.md)
        p.parent.mkdir(parents=True, exist_ok=True)
        md = render_md(rows, args, notes)
        p.write_text(md + "\n" + "\n".join(f"> {n}" for n in notes) + "\n", encoding="utf-8")
        print(f"  Markdown -> {p}")


if __name__ == "__main__":
    main()
