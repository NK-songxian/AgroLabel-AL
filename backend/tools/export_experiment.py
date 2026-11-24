"""导出一轮三臂实验结果，并计算与真值的一致性/准确率。

产出（默认写到 `out/report/exp_<时间戳>/`）：

    per_image.csv  每个 (图, 臂) 一行：GT 实例数、提交实例数、TP/FP/FN、
                   precision / recall / F1、类别正确率、掩膜 IoU / Dice、
                   两个口径的耗时、四类操作次数
    per_arm.csv    按臂聚合：均值/中位、耗时中位、总操作数
    paired.csv     同图配对（A−B 的耗时差、IoU 差、F1 差）
    summary.md     人读的报告（含方法说明与局限）

口径
----
- 实例匹配：贪心按 IoU 从大到小配，要求 IoU ≥ --iou（默认 0.5）；
  先算**类别无关**的 TP/FP/FN（几何检测能力），再算**同类匹配**
  （类别正确率 = 同类匹配 / 全部匹配）
- 掩膜 IoU/Dice：把该图所有实例并集画成掩膜再比（衡量整体覆盖）
- 光栅化：按原图长宽比缩放到长边 --maskpx（默认 512），避免非等比缩放改变面积比
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

ARMS = ("assisted", "manual", "noprompt")
ARM_LABEL = {
    "assisted": "预标 + 点提示",
    "manual": "纯人工",
    "noprompt": "预标 · 禁用点提示",
}
OPS_KEYS = ("新画", "删实例", "改类别", "拖边界")


# ---------------------------------------------------------------- 读文件
def read_yolo_seg(path: Path) -> list[tuple[int, list[tuple[float, float]]]]:
    """YOLO-seg：每行 `class_id x1 y1 x2 y2 ...`（归一化）。"""
    if not path.is_file():
        return []
    out: list[tuple[int, list[tuple[float, float]]]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            cid = int(parts[0])
            nums = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(nums) % 2:
            nums = nums[:-1]
        poly = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
        if len(poly) >= 3:
            out.append((cid, poly))
    return out


def image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return (0, 0)


# ---------------------------------------------------------------- 光栅化
def canvas_size(w: int, h: int, long_edge: int) -> tuple[int, int]:
    if w <= 0 or h <= 0:
        return (long_edge, long_edge)
    s = long_edge / max(w, h)
    return (max(1, round(w * s)), max(1, round(h * s)))


def raster(
    items: list[tuple[int, list[tuple[float, float]]]],
    size: tuple[int, int],
) -> list[Image.Image]:
    """每个实例一张二值掩膜（L 模式，255=前景）。"""
    cw, ch = size
    out = []
    for _, poly in items:
        im = Image.new("L", (cw, ch), 0)
        pts = [(x * cw, y * ch) for x, y in poly]
        if len(pts) >= 3:
            ImageDraw.Draw(im).polygon(pts, fill=255)
        out.append(im)
    return out


def union_mask(masks: list[Image.Image], size: tuple[int, int]) -> Image.Image:
    acc = Image.new("L", size, 0)
    for m in masks:
        # 逐像素取 max = 并集
        from PIL import ImageChops

        acc = ImageChops.lighter(acc, m)
    return acc


def mask_iou_dice(a: Image.Image, b: Image.Image) -> tuple[float, float]:
    from PIL import ImageChops

    sa = a.histogram()[255]
    sb = b.histogram()[255]
    inter = ImageChops.darker(a, b).histogram()[255]
    union = sa + sb - inter
    iou = inter / union if union else (1.0 if sa == sb == 0 else 0.0)
    dice = 2 * inter / (sa + sb) if (sa + sb) else (1.0 if union == 0 else 0.0)
    return iou, dice


def inst_iou(a: Image.Image, b: Image.Image) -> float:
    from PIL import ImageChops

    inter = ImageChops.darker(a, b).histogram()[255]
    sa = a.histogram()[255]
    sb = b.histogram()[255]
    union = sa + sb - inter
    return inter / union if union else 0.0


# ---------------------------------------------------------------- 匹配
def match_instances(
    gt: list[tuple[int, list[tuple[float, float]]]],
    pred: list[tuple[int, list[tuple[float, float]]]],
    size: tuple[int, int],
    thr: float,
) -> dict:
    g_masks = raster(gt, size)
    p_masks = raster(pred, size)
    pairs = []
    for gi, gm in enumerate(g_masks):
        for pi, pm in enumerate(p_masks):
            iou = inst_iou(gm, pm)
            if iou >= thr:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True)

    used_g, used_p = set(), set()
    same_class = 0
    for _iou, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        if gt[gi][0] == pred[pi][0]:
            same_class += 1

    tp = len(used_g)
    fp = len(pred) - tp
    fn = len(gt) - tp
    p = tp / (tp + fp) if (tp + fp) else (0.0 if fn else 1.0)
    r = tp / (tp + fn) if (tp + fn) else (1.0 if not fp else 0.0)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": p, "recall": r, "f1": f1,
        "matched": tp, "same_class": same_class,
        "class_acc": (same_class / tp) if tp else None,
        "gt_masks": g_masks, "pred_masks": p_masks,
    }


# ---------------------------------------------------------------- 耗时/操作
def same_as_pred(
    pred_snapshot: list,
    annot: list[tuple[int, list[tuple[float, float]]]],
    tol: float = 1e-4,
) -> bool:
    """提交内容是否与预标注完全一致（= 标注员其实没改）。

    关键诊断：若 assisted/noprompt 的提交与预标逐顶点相同，那这条的 F1/IoU
    反映的是**模型**精度，不能当作"人机协同后的精度"。
    """
    if not isinstance(pred_snapshot, list) or len(pred_snapshot) != len(annot):
        return False
    for snap, (cid, poly) in zip(pred_snapshot, annot, strict=False):
        if int(snap.get("class_id", -1)) != cid:
            return False
        sp = snap.get("polygon") or []
        if len(sp) != len(poly):
            return False
        for (x1, y1), (x2, y2) in zip(sp, poly, strict=False):
            if abs(float(x1) - x2) > tol or abs(float(y1) - y2) > tol:
                return False
    return True


def load_sidecar(annot_dir: Path, arm: str, image_id: str) -> dict:
    p = annot_dir / arm / f"{image_id}.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_ops(log_dir: Path) -> dict[str, Counter]:
    """arm -> Counter(image -> Counter(op))"""
    out: dict[str, Counter] = {}
    for arm in ARMS:
        f = log_dir / f"ops_{arm}.csv"
        c: Counter = Counter()
        if f.is_file():
            with f.open(encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    key = (row.get("image") or "", row.get("op") or "")
                    if key[0]:
                        c[key] += 1
        out[arm] = c
    return out


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="导出三臂实验结果 + 真值准确率")
    ap.add_argument("--manifest", default="out/experiment/manifest.json")
    ap.add_argument("--annotations", default="out/annotations")
    ap.add_argument("--logs", default="out/logs")
    ap.add_argument("--data-root", default="C:/Users/1/Desktop/附件/AL_v4/dataset")
    ap.add_argument("--split", default="val")
    ap.add_argument("--iou", type=float, default=0.5, help="实例匹配的 IoU 阈值")
    ap.add_argument("--maskpx", type=int, default=512, help="光栅化长边像素")
    ap.add_argument("--group", default=None, help="只导出某一组（如 g1）")
    ap.add_argument("--exclude-image", nargs="*", default=[],
                    help="排除这些图（例如怀疑看过真值的那张），做敏感性分析")
    ap.add_argument("--out", default=None, help="输出目录（默认 out/report/exp_<时间戳>）")
    args = ap.parse_args()

    ws = Path(__file__).resolve().parents[2]
    man_path = Path(args.manifest)
    man_path = man_path if man_path.is_absolute() else ws / man_path
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    annot_dir = Path(args.annotations)
    annot_dir = annot_dir if annot_dir.is_absolute() else ws / annot_dir
    log_dir = Path(args.logs)
    log_dir = log_dir if log_dir.is_absolute() else ws / log_dir
    data_root = Path(args.data_root)
    gt_dir = data_root / "labels" / args.split
    img_dir = data_root / "images" / args.split

    ops = load_ops(log_dir)
    rows: list[dict] = []
    skipped: list[str] = []
    for t in manifest["tasks"]:
        if args.group and t["group"] != args.group:
            continue
        arm, iid = t["arm"], t["image_id"]
        if iid in set(args.exclude_image):
            continue
        annot = read_yolo_seg(annot_dir / arm / f"{iid}.txt")
        side = load_sidecar(annot_dir, arm, iid)
        if not side and not annot:
            skipped.append(f"{arm}/{iid}")
            continue
        gt = read_yolo_seg(gt_dir / f"{iid}.txt")
        w, h = image_size(img_dir / f"{iid}.jpg")
        size = canvas_size(w, h, args.maskpx)

        m = match_instances(gt, annot, size, args.iou)
        u_gt = union_mask(m["gt_masks"], size)
        u_pr = union_mask(m["pred_masks"], size)
        miou, mdice = mask_iou_dice(u_gt, u_pr)

        c = ops.get(arm, Counter())
        unchanged = same_as_pred(side.get("pred_snapshot") or [], annot)
        rows.append({
            "group": t["group"], "arm": arm, "image_id": iid,
            "class_name": t.get("class_name"),
            "label_state": side.get("label_state"),
            "n_gt": len(gt), "n_pred": len(annot),
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "class_acc": (round(m["class_acc"], 4) if m["class_acc"] is not None else ""),
            "mask_iou": round(miou, 4), "mask_dice": round(mdice, 4),
            "elapsed_ms": side.get("elapsed_ms"),
            "elapsed_from_open_ms": side.get("elapsed_from_open_ms"),
            "ops_new": c[(iid, "新画")], "ops_del": c[(iid, "删实例")],
            "ops_class": c[(iid, "改类别")], "ops_vertex": c[(iid, "拖边界")],
            # 只对 assisted/noprompt 有意义；manual 没有预标，留空
            "same_as_pred": ("" if arm == "manual" else int(bool(unchanged))),
        })

    if not rows:
        print("没有可导出的记录（检查 --group / 清单 / annotations 路径）")
        return 2

    out_dir = Path(args.out) if args.out else (
        ws / "out" / "report" / f"exp_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- per_image.csv
    fields = list(rows[0].keys())
    with (out_dir / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---- per_arm.csv
    def agg(arm: str) -> dict:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            return {}
        ms = [r["mask_iou"] for r in rs]
        f1s = [r["f1"] for r in rs]
        ps = [r["precision"] for r in rs]
        rcs = [r["recall"] for r in rs]
        cls = [r["class_acc"] for r in rs if r["class_acc"] != ""]
        t_ms = [r["elapsed_ms"] for r in rs if isinstance(r["elapsed_ms"], (int, float))]
        t_open = [r["elapsed_from_open_ms"] for r in rs
                  if isinstance(r["elapsed_from_open_ms"], (int, float))]
        tp_s = sum(r["tp"] for r in rs)
        fp_s = sum(r["fp"] for r in rs)
        fn_s = sum(r["fn"] for r in rs)
        p_micro = tp_s / (tp_s + fp_s) if (tp_s + fp_s) else 0.0
        r_micro = tp_s / (tp_s + fn_s) if (tp_s + fn_s) else 0.0
        f1_micro = (2 * p_micro * r_micro / (p_micro + r_micro)) if (p_micro + r_micro) else 0.0
        return {
            "arm": arm,
            "precision_micro": round(p_micro, 4),
            "recall_micro": round(r_micro, 4),
            "f1_micro": round(f1_micro, 4), "arm_label": ARM_LABEL[arm], "n_images": len(rs),
            "same_as_pred": sum(
                1 for r in rs if r["same_as_pred"] == 1
            ),
            "n_gt": sum(r["n_gt"] for r in rs), "n_pred": sum(r["n_pred"] for r in rs),
            "tp": sum(r["tp"] for r in rs), "fp": sum(r["fp"] for r in rs),
            "fn": sum(r["fn"] for r in rs),
            "precision_macro": round(statistics.mean(ps), 4),
            "recall_macro": round(statistics.mean(rcs), 4),
            "f1_macro": round(statistics.mean(f1s), 4),
            "f1_median": round(statistics.median(f1s), 4),
            "mask_iou_mean": round(statistics.mean(ms), 4),
            "mask_iou_median": round(statistics.median(ms), 4),
            "class_acc_mean": (round(statistics.mean(cls), 4) if cls else ""),
            "elapsed_ms_median": (round(statistics.median(t_ms)) if t_ms else ""),
            "elapsed_open_median": (round(statistics.median(t_open)) if t_open else ""),
            "ops_new": sum(r["ops_new"] for r in rs),
            "ops_del": sum(r["ops_del"] for r in rs),
            "ops_class": sum(r["ops_class"] for r in rs),
            "ops_vertex": sum(r["ops_vertex"] for r in rs),
        }

    arm_rows = [a for a in (agg(x) for x in ARMS) if a]
    with (out_dir / "per_arm.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(arm_rows[0].keys()))
        w.writeheader()
        w.writerows(arm_rows)

    # ---- paired.csv（同图配对）
    by_img: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_img.setdefault(r["image_id"], {})[r["arm"]] = r
    pairs = [("assisted", "manual"), ("assisted", "noprompt"), ("manual", "noprompt")]
    pair_rows = []
    for a, b in pairs:
        common = [v for v in by_img.values() if a in v and b in v]
        if not common:
            continue
        d_ms = []
        d_open = []
        d_iou = []
        d_f1 = []
        for v in common:
            ra, rb = v[a], v[b]
            for key, bucket in (("elapsed_ms", d_ms), ("elapsed_from_open_ms", d_open)):
                x, y = ra[key], rb[key]
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    bucket.append(x - y)
            d_iou.append(ra["mask_iou"] - rb["mask_iou"])
            d_f1.append(ra["f1"] - rb["f1"])
        pair_rows.append({
            "pair": f"{a} - {b}", "n_pairs": len(common),
            "median_d_elapsed_ms": round(statistics.median(d_ms)) if d_ms else "",
            "median_d_elapsed_open_ms": round(statistics.median(d_open)) if d_open else "",
            "a_faster_count": sum(1 for x in d_open if x < 0),
            "median_d_mask_iou": round(statistics.median(d_iou), 4),
            "median_d_f1": round(statistics.median(d_f1), 4),
        })
    if pair_rows:
        with (out_dir / "paired.csv").open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys()))
            w.writeheader()
            w.writerows(pair_rows)

    # ---- summary.md
    groups = sorted({r["group"] for r in rows})
    lines = [
        "# 三臂实验结果（真值一致性）",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 清单：`{man_path.name}`（{manifest.get('name')}）",
        f"- 范围：{'、'.join(groups)} · {len(rows)} 条（{len(by_img)} 张图）",
        f"- 匹配口径：实例 IoU ≥ {args.iou}，光栅化长边 {args.maskpx}px",
        "",
        "## 每臂指标",
        "",
        "| 臂 | 图数 | 未改动 | GT | 提交 | P(宏) | R(宏) | F1(宏) | P(微) | R(微) | F1(微) | 掩膜IoU(均) | 类别正确率 | 耗时中位(首动作) | 耗时中位(开图) | 新画 | 删 | 改类 | 拖边界 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in arm_rows:
        lines.append(
            f"| {a['arm_label']} | {a['n_images']} | {a['same_as_pred']}/{a['n_images']} "
            f"| {a['n_gt']} | {a['n_pred']} "
            f"| {a['precision_macro']:.3f} | {a['recall_macro']:.3f} | {a['f1_macro']:.3f} "
            f"| {a['precision_micro']:.3f} | {a['recall_micro']:.3f} | {a['f1_micro']:.3f} "
            f"| {a['mask_iou_mean']:.3f} | {a['class_acc_mean'] if a['class_acc_mean'] != '' else '—'} "
            f"| {a['elapsed_ms_median'] or '—'} ms | {a['elapsed_open_median'] or '—'} ms "
            f"| {a['ops_new']} | {a['ops_del']} | {a['ops_class']} | {a['ops_vertex']} |"
        )
    if pair_rows:
        lines += [
            "",
            "## 同图配对（A − B）",
            "",
            "负值表示 A 更省时间 / 更准。",
            "",
            "| 配对 | 对数 | 耗时差中位(首动作) | 耗时差中位(开图) | A更快的对数 | IoU 差中位 | F1 差中位 |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in pair_rows:
            lines.append(
                f"| {r['pair']} | {r['n_pairs']} | {r['median_d_elapsed_ms']} ms "
                f"| {r['median_d_elapsed_open_ms']} ms | {r['a_faster_count']}/{r['n_pairs']} "
                f"| {r['median_d_mask_iou']:+.3f} | {r['median_d_f1']:+.3f} |"
            )
    lines += [
        "",
        "## 方法说明",
        "",
        "- **P/R/F1**：实例级贪心匹配（IoU≥阈值），先按几何匹配算 TP/FP/FN；类别正确率单独统计",
        "- **掩膜 IoU/Dice**：该图全部实例并集 vs GT 并集，衡量整体覆盖与形状",
        "- **耗时**：`elapsed_ms` = 首个标注动作→提交；`elapsed_from_open_ms` = 开图→提交（主口径）",
        "- **操作次数**：来自 `out/logs/ops_<arm>.csv` 的四类操作计数",
        "",
        "## 局限",
        "",
        "- 【未改动】列很关键：assisted/noprompt 若有大量条目未改动，说明标注员没有编辑，",
        "  这些条的精度反映的是模型本身，不能当作人机协同结果",
        "- 单组 8 图/臂，指标波动大；结论需 4 组跑完后再看组间一致性",
        "- GT 本身有多边形简化误差；IoU 阈值 0.5 对细小病斑偏宽松",
        "- 同一标注员同一批图重复标注，存在记忆效应（方向偏保守）",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "manifest_snapshot.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"  导出目录: {out_dir}")
    print(f"  记录 {len(rows)} 条 / {len(by_img)} 张图" + (f"；跳过 {len(skipped)}" if skipped else ""))
    print("\n  每臂:")
    for a in arm_rows:
        print(f"    {a['arm_label']:16} P={a['precision_macro']:.3f} R={a['recall_macro']:.3f} "
              f"F1={a['f1_macro']:.3f} | 微 P={a['precision_micro']:.3f} "
              f"R={a['recall_micro']:.3f} F1={a['f1_micro']:.3f} "
              f"| IoU={a['mask_iou_mean']:.3f} "
              f"耗时中位={a['elapsed_ms_median']}ms 开图={a['elapsed_open_median']}ms")
    if pair_rows:
        print("\n  同图配对:")
        for r in pair_rows:
            print(f"    {r['pair']:24} n={r['n_pairs']:2} "
                  f"Δ耗时={r['median_d_elapsed_ms']:>6}ms ΔIoU={r['median_d_mask_iou']:+.3f} "
                  f"ΔF1={r['median_d_f1']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
