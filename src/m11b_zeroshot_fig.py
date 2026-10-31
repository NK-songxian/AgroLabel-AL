# -*- coding: utf-8 -*-
"""
m11b_zeroshot_fig.py —— 把零样本探针的测量结果画成一张能直接进报告的图

读 out/zeroshot_probe.json，不重新推理。
左图：逐图"最高置信度"分布 —— 零样本 vs 训练后，同一批 40 张 test 图
右图：conf=0.25（可用阈值）下的三项可用性指标
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

d = json.loads((OUT / "zeroshot_probe.json").read_text(encoding="utf-8"))

zs = d["prompt_sets"]["coarse-5"]["by_conf"]["conf0.001"]["max_confs_per_image"]
zs_leaf = d["prompt_sets"]["leaf-1"]["by_conf"]["conf0.001"]["max_confs_per_image"]
tr = d["trained_reference"]["max_confs_per_image"]

fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8),
                         gridspec_kw={"width_ratios": [1.25, 1]})

# ---------------- 左：置信度分布 ----------------
ax = axes[0]
bins = np.linspace(0, 1.0, 26)
ax.hist(tr, bins=bins, color="#2e7d32", alpha=0.75,
        label=f"训练后 YOLO11n-seg（中位数 {np.median(tr):.3f}）", edgecolor="white")
ax.hist(zs, bins=bins, color="#c62828", alpha=0.85,
        label=f"零样本 YOLO-World·病害提示（中位数 {np.median(zs):.4f}）", edgecolor="white")
ax.hist(zs_leaf, bins=bins, color="#ef6c00", alpha=0.55,
        label=f"零样本 YOLO-World·leaf 提示（中位数 {np.median(zs_leaf):.4f}）", edgecolor="white")
ax.axvline(0.25, color="#1565c0", lw=2, ls="--")
ax.text(0.265, ax.get_ylim()[1] * 0.92, "可用阈值 conf=0.25", color="#1565c0", fontsize=11)
ax.set_xlim(0, 1.0)
ax.set_xlabel("该图所有检测框中的最高置信度")
ax.set_ylabel("图像数")
ax.set_title("同一批 40 张 PlantSeg test 图：最高置信度分布", fontsize=13, pad=10)
ax.legend(fontsize=10, loc="upper right")
ax.grid(alpha=0.25, axis="y")
ax.set_axisbelow(True)
ax.text(0.28, 0.58,
        f"零样本 40/40 张图的最高分\n都落在 conf<0.11 的噪声带内",
        transform=ax.transAxes, fontsize=11, color="#c62828",
        bbox=dict(boxstyle="round,pad=0.45", fc="#fff3f3", ec="#c62828", alpha=0.92))

# ---------------- 右：可用性三项 ----------------
ax = axes[1]
op = d["op_conf"]
zs_op = d["prompt_sets"]["coarse-5"]["by_conf"][f"conf{op}"]
key_zs_001 = d["prompt_sets"]["coarse-5"]["by_conf"]["conf0.001"]
trr = d["trained_reference"]

metrics = ["出框图占比", "框落在病斑上", "GT 实例被覆盖"]
zero_vals = [zs_op["img_det_rate"], zs_op["box_on_gt_rate"], zs_op["gt_instance_coverage"]]
train_vals = [trr["img_det_rate"], trr["box_on_gt_rate"], trr["gt_instance_coverage"]]
noise_vals = [key_zs_001["img_det_rate"], key_zs_001["box_on_gt_rate"],
              key_zs_001["gt_instance_coverage"]]

x = np.arange(len(metrics))
w = 0.27
b1 = ax.bar(x - w, zero_vals, w, label=f"零样本（conf={op}）", color="#c62828")
b2 = ax.bar(x, noise_vals, w, label="零样本（conf=0.001，纯噪声）", color="#ef9a9a")
b3 = ax.bar(x + w, train_vals, w, label=f"训练后（conf={op}）", color="#2e7d32")

for bars in (b1, b2, b3):
    for r in bars:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h + 0.018, f"{h:.0%}",
                ha="center", fontsize=10.5)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11.5)
ax.set_ylim(0, 1.12)
ax.set_ylabel("比例")
ax.set_title("可用性对比：零样本 vs 训练后（40 张图 / 174 个真实病斑）", fontsize=13, pad=10)
ax.legend(fontsize=10, loc="upper left")
ax.grid(alpha=0.25, axis="y")
ax.set_axisbelow(True)

fig.suptitle("零样本预标注可行性实测：开放词表检测器在细粒度病斑上给不出可用框",
             fontsize=15, y=0.99)
fig.text(0.99, 0.012,
         "注：'GT 实例被覆盖'为「框中心落入真值掩码」的宽松代理指标，不等于 IoU≥0.5 的严格实例召回；"
         "管线自检：同代码同权重在官方样例 bus.jpg 上给出 bus=0.909 / person=0.908，故零样本的低分不是配置问题。",
         ha="right", va="bottom", fontsize=8.6, color="#555")
fig.tight_layout(rect=[0, 0.045, 1, 0.94])
dst = FIG / "zeroshot_vs_trained.png"
fig.savefig(dst, dpi=170)
print(f"[输出] {dst}")
