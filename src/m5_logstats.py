"""M5 把人机协同的操作日志算成验收要的数字。

验收里那条「同精度下单张修改次数下降 ≥30%」不是一句形容词，
得能算出两个数来比。口径定死在这里：

  单张修改次数 = 该图提交前，人为改动的次数
                （改类别 / 删实例 / 标为无目标 各算 1 次；
                  「整张确认」算 0 次，因为它表示一个字都没改；
                  「撤销」不单独计数，只把前面那次抵消掉）

对照组 = 同一批图，人工从空白开始标（相当于 conf 阈值调到 1.0 的模拟）。
处理组 = 模型预标注 + 人工修正。

运行：
  python src/m5_logstats.py
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from common import load_config, setup_matplotlib

COUNTED = {"改类别", "删实例", "标为无目标"}
NEUTRAL = {"整张确认", "撤销", "提交"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT / "configs" / "default.yaml"))
    ap.add_argument("--ops", default=None,
                    help="ops.csv 路径；三臂对照时传 out/logs/ops_<arm>.csv 分别统计")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ops_path = Path(args.ops) if args.ops else PROJECT / "out" / "logs" / "ops.csv"
    if not ops_path.exists():
        raise SystemExit(f"还没有操作日志：{ops_path}\n先用 m4_app.py 标几张再来")

    rows = list(csv.DictReader(open(ops_path, encoding="utf-8-sig")))
    if not rows:
        raise SystemExit("日志是空的")

    per_image = defaultdict(int)
    submitted, op_kind = set(), Counter()
    api_ms = []
    for r in rows:
        op_kind[r["op"]] += 1
        if r["api_ms"]:
            try:
                api_ms.append(float(r["api_ms"]))
            except ValueError:
                pass
        if r["op"] in COUNTED:
            per_image[r["image"]] += 1
        elif r["op"] == "提交":
            submitted.add(r["image"])

    # 只统计真正提交过的图，没提交的是半成品，不能进分母
    counts = np.array([per_image.get(k, 0) for k in sorted(submitted)], dtype=float)
    if counts.size == 0:
        raise SystemExit("还没有已提交的图，无法统计")

    print("=== 单张修改次数 ===")
    print(f"  已提交 {len(counts)} 张")
    print(f"  均值 {counts.mean():.2f} 次/张   中位 {np.median(counts):.0f}")
    print(f"  分布：零改动 {int((counts == 0).sum())} 张 "
          f"({(counts == 0).mean()*100:.0f}%)，"
          f"改动 ≥1 次 {int((counts >= 1).sum())} 张")
    print(f"  最多改了 {int(counts.max())} 次")
    print("\n=== 操作类型 ===")
    for k, v in op_kind.most_common():
        print(f"  {k:<8}{v:>6}")

    if api_ms:
        a = np.array(api_ms)
        print("\n=== 预标注时延 ===")
        print(f"  中位 {np.median(a):.0f} ms   P90 {np.percentile(a, 90):.0f} ms"
              f"   （验收线 300 ms）")

    fig_dir = PROJECT / cfg["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(counts, bins=range(0, int(counts.max()) + 2), color="#4a7ebb",
               edgecolor="white")
    ax[0].axvline(counts.mean(), color="#c0392b", ls="--",
                  label=f"均值 {counts.mean():.1f}")
    ax[0].set_xlabel("单张修改次数")
    ax[0].set_ylabel("图片数")
    ax[0].set_title("预标注后的人工修改量分布")
    ax[0].legend()

    kinds = [k for k, _ in op_kind.most_common()]
    ax[1].barh(kinds[::-1], [op_kind[k] for k in kinds[::-1]], color="#5aa469")
    ax[1].set_xlabel("事件数")
    ax[1].set_title("操作类型构成")
    plt.tight_layout()
    out = fig_dir / "ops_stats.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n图：{out}")


if __name__ == "__main__":
    main()
