"""把训练日志里的逐类指标拆成「头部 vs 尾部」，产出小样本表现的证据表。

12 分那一条考的是「细分类别/小样本表现」，光给一个总的 mAP 是拿不到分的。
要回答三个问题：

  1. 长尾到底有多长 —— 各类图像数的分布、最长/最短之比
  2. 尾巴上掉了多少 —— 头部 50% 与尾部 50% 的 mAP 分别是多少，差值多少
  3. 为什么掉 —— 每类图像数与 mAP 的相关性、样本数低于阈值时的表现

产出：
  out/report/longtail_table.csv    逐类明细（样本数、占比、mAP50、是否在尾部队列）
  out/report/longtail_summary.json 头部/尾部对比与相关性
  out/figures/longtail.png         样本数-精度散点 + 分组柱状图

用法：
  python src/m7_longtail.py --metrics out/report/plantseg115_metrics.json \
      --data H:/Plant/_derived/dataset_plantseg_115
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from common import setup_matplotlib  # noqa: E402


def count_per_class(data_root, split="train"):
    """数每个类别在训练集里出现在多少张图上。

    注意数的是**图片数**，不是标签行数：一张图上可能有 7 个同类病斑，
    那是 7 条标签但只算 1 张图。按标签行数数会把长尾比虚高
    （实测按行数算是 37:1，按图数算是 29:1）。
    """
    from collections import defaultdict
    seen = defaultdict(set)
    lab_dir = Path(data_root) / "labels" / split
    if not lab_dir.is_dir():
        raise SystemExit(f"找不到 {lab_dir}")
    for p in lab_dir.glob("*.txt"):
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split(maxsplit=1)
            if parts:
                try:
                    seen[int(parts[0])].add(p.stem)
                except ValueError:
                    pass
    return {k: len(v) for k, v in seen.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="m3_train.py 产出的 *_metrics.json")
    ap.add_argument("--data", required=True, help="派生数据集根目录（含 classes.json）")
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="用哪个划分的 mAP 做对比")
    ap.add_argument("--slot", default=None,
                    help="给这张表起个名字，用于输出文件名；默认取 metrics 文件名")
    args = ap.parse_args()

    info = json.loads((Path(args.data) / "classes.json").read_text(encoding="utf-8"))
    classes = info["classes"]
    m = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    node = m.get(args.split) or m
    per = node.get("per_class_mask_mAP50") or {}
    if not per:
        raise SystemExit("指标文件里没有 per_class_mask_mAP50，先重跑 m3_train.py")

    cnt = count_per_class(args.data, "train")
    rows = []
    for i, name in enumerate(classes):
        rows.append({"cls_id": i, "class": name,
                     "train_images": cnt.get(i, 0),
                     "mask_mAP50": float(per.get(name, 0.0))})
    rows.sort(key=lambda r: -r["train_images"])

    n = len(rows)
    half = n // 2
    head, tail = rows[:half], rows[half:]
    head_imgs = sum(r["train_images"] for r in head)
    tot_imgs = sum(r["train_images"] for r in rows) or 1

    def agg(g):
        v = np.array([r["mask_mAP50"] for r in g], dtype=float)
        return {"n_classes": len(g),
                "mean_mAP50": round(float(v.mean()), 4),
                "median_mAP50": round(float(np.median(v)), 4),
                "zero_classes": int((v < 1e-6).sum()),
                "min_images": int(min(r["train_images"] for r in g)),
                "max_images": int(max(r["train_images"] for r in g))}

    x = np.array([r["train_images"] for r in rows], dtype=float)
    y = np.array([r["mask_mAP50"] for r in rows], dtype=float)
    # 用 log 样本数算相关：样本数跨两个数量级，线性相关会被头部几类带跑
    corr_log = float(np.corrcoef(np.log10(np.maximum(x, 1)), y)[0, 1]) if n > 2 else 0.0

    # 短样本类单独看一眼：训练图少于 20 张的类
    few = [r for r in rows if r["train_images"] < 20]
    few_agg = {"threshold": 20, "n_classes": len(few),
               "mean_mAP50": round(float(np.mean([r["mask_mAP50"] for r in few])), 4)
               if few else 0.0}

    summary = {
        "slot": args.slot or Path(args.metrics).stem,
        "class_mode": info.get("mode"),
        "split": args.split,
        "n_classes": n,
        "train_images_total": tot_imgs,
        "long_tail_ratio": round(max(x) / max(min(x), 1), 1),
        "median_images_per_class": int(np.median(x)),
        "head": agg(head), "tail": agg(tail),
        "tail_share_of_train_images": round(
            sum(r["train_images"] for r in tail) / tot_imgs, 3),
        "gap_head_minus_tail": round(agg(head)["mean_mAP50"] - agg(tail)["mean_mAP50"], 4),
        "corr_log_images_vs_mAP": round(corr_log, 3),
        "few_shot": few_agg,
        "overall": {"mask_mAP50": node.get("mask_mAP50"),
                    "mask_mAP50_95": node.get("mask_mAP50_95"),
                    "box_mAP50": node.get("box_mAP50")},
    }

    out = PROJECT / "out" / "report"
    out.mkdir(parents=True, exist_ok=True)
    tag = summary["slot"]
    with open(out / f"longtail_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out / f"longtail_{tag}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "cls_id", "class", "train_images",
                                          "mask_mAP50", "bucket"])
        w.writeheader()
        for k, r in enumerate(rows, 1):
            w.writerow({"rank": k, **r,
                        "bucket": "head" if k <= half else "tail"})

    print(f"类别空间 {summary['class_mode']}，{n} 类，训练图 {tot_imgs}")
    print(f"长尾比 {summary['long_tail_ratio']}:1，每类中位数 "
          f"{summary['median_images_per_class']} 张")
    print(f"头部 {half} 类占训练图 {1 - summary['tail_share_of_train_images']:.1%}，"
          f"平均 mAP50 {summary['head']['mean_mAP50']:.4f}")
    print(f"尾部 {n - half} 类平均 mAP50 {summary['tail']['mean_mAP50']:.4f}，"
          f"差值 {summary['gap_head_minus_tail']:+.4f}")
    print(f"完全没学到（mAP=0）的类：头 {summary['head']['zero_classes']} / "
          f"尾 {summary['tail']['zero_classes']}")
    print(f"少于 {few_agg['threshold']} 张的类 {few_agg['n_classes']} 个，"
          f"平均 mAP50 {few_agg['mean_mAP50']:.4f}")
    print(f"log(样本数) 与 mAP50 的相关系数 {corr_log:+.3f}")

    # ---------------------------------------------------------- 图
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.scatter(x, y, s=22, c="#c0392b", alpha=0.7, edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("训练集中该类出现的图片数（log）")
    ax.set_ylabel("该类掩码 mAP@0.5")
    ax.set_title(f"样本数 vs 精度（{n} 类，r={corr_log:+.2f}）")
    ax.grid(alpha=0.3, linestyle="--")

    ax = axes[1]
    labels = ["头部 50%", "尾部 50%"]
    vals = [summary["head"]["mean_mAP50"], summary["tail"]["mean_mAP50"]]
    bars = ax.bar(labels, vals, color=["#2e86c1", "#e67e22"], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", fontsize=11)
    ax.set_ylabel("平均掩码 mAP@0.5")
    ax.set_title(f"头部与尾部平均精度（差 {summary['gap_head_minus_tail']:+.3f}）")
    ax.set_ylim(0, max(vals) * 1.35 + 0.02)
    ax.grid(alpha=0.3, axis="y", linestyle="--")

    plt.tight_layout()
    figd = PROJECT / "out" / "figures"
    figd.mkdir(parents=True, exist_ok=True)
    plt.savefig(figd / f"longtail_{tag}.png", dpi=150)
    plt.close(fig)
    print(f"\n表：{out / f'longtail_{tag}.csv'}")
    print(f"汇总：{out / f'longtail_{tag}.json'}")
    print(f"图：{figd / f'longtail_{tag}.png'}")


if __name__ == "__main__":
    main()
