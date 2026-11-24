# -*- coding: utf-8 -*-
"""正式 AL 消融可视化：4 策略 x 5 轮（2026-11-24 云端完成）。

输入：runtime/al_full_ablation_curve.csv（从云端 out/report/ 下载）
输出：docs/al_effect_full.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(
    (ROOT / 'runtime/al_full_ablation_curve.csv').open(encoding='utf-8-sig')))

series = {}
for r in rows:
    s = r['strategy']
    series.setdefault(s, {'n': [], 'p50': [], 'p5095': []})
    series[s]['n'].append(int(r['n_labeled']))
    series[s]['p50'].append(float(r['mask_mAP50']))
    series[s]['p5095'].append(float(r['mask_mAP50_95']))

COLORS = {'random': '#7f8c8d', 'entropy': '#2980b9',
          'tta': '#c0392b', 'coreset': '#27ae60'}
LABELS = {'random': 'random（随机对照）', 'entropy': 'entropy（不确定性）',
          'tta': 'TTA（多尺度不确定性）', 'coreset': 'coreset（多样性）'}

B6 = 0.416          # 全监督 B6，5367 张，ultralytics val 口径
TARGET = 0.2235     # 最好成绩的 90%，脚本定义

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

# ---- 左：4 策略采样效率曲线 ----
for s, d in series.items():
    ax1.plot(d['n'], d['p50'], 'o-', color=COLORS[s], lw=2.1, ms=6,
             label=LABELS[s])
ax1.axhline(B6, ls='--', color='#2c3e50', lw=1.5,
            label=f'B6 全监督参考 {B6}（5367 张）')
ax1.axhline(TARGET, ls=':', color='#e67e22', lw=1.6,
            label=f'目标线 {TARGET}（最优成绩 90%）')
ax1.annotate('TTA r4=0.2356 达标\n(1786 张)',
             xy=(1786, 0.2356), xytext=(1180, 0.27),
             arrowprops=dict(arrowstyle='->', color=COLORS['tta']),
             fontsize=9.5, color=COLORS['tta'])
ax1.set_xlabel('已标注图像数', fontsize=11)
ax1.set_ylabel('test mask mAP@0.5', fontsize=11)
ax1.set_title('4 策略采样效率曲线（25ep/轮，同配置）', fontsize=12)
ax1.legend(fontsize=8.5, loc='lower right')
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 0.47)

# ---- 右：终轮对比柱状图 ----
finals = {s: d['p50'][-1] for s, d in series.items()}
names = list(finals)
vals = [finals[s] for s in names]
bars = ax2.bar([LABELS[s] for s in names], vals,
               color=[COLORS[s] for s in names], width=0.58)
for b, v in zip(bars, vals):
    ax2.annotate(f'{v:.4f}', (b.get_x() + b.get_width()/2, v),
                 ha='center', va='bottom', fontsize=10)
ax2.axhline(B6, ls='--', color='#2c3e50', lw=1.4, label=f'B6 全监督 {B6}')
ax2.set_ylabel('test mask mAP@0.5（r5，2207 张）', fontsize=11)
ax2.set_title('同预算终轮对比：TTA 领先 random +18.1%', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 0.47)
ax2.tick_params(axis='x', labelsize=9)

fig.suptitle('AgroLabel-AL · 主动学习四策略消融（2026-11-24 云端 96G 单卡完成）',
             fontsize=13, y=1.02)
fig.tight_layout()
out = ROOT / 'docs/al_effect_full.png'
fig.savefig(out, dpi=160, bbox_inches='tight')
print('saved:', out)
