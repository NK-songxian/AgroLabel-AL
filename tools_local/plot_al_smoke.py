# -*- coding: utf-8 -*-
"""AL 冒烟效果可视化：B6 配方（YOLO26x-seg@1280）主动学习闭环曲线。

数据：2026-11-17 云端冒烟（random 策略，3 轮，15 epochs/轮），
test split（1560 图）ultralytics 口径。与全监督基线同画布对比。
输出：docs/al_effect_smoke.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 冒烟实测（random 策略）
n_labeled = [102, 523, 944]
mAP50 = [0.0154, 0.0604, 0.1094]

B5, B6 = 0.397, 0.416

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

# ---- 左：AL 增长曲线 + 全监督参考线 ----
ax1.plot(n_labeled, mAP50, 'o-', color='#c0392b', lw=2.2, ms=8,
         label='AL 闭环实测（random，B6 配方 @1280）')
for x, y in zip(n_labeled, mAP50):
    ax1.annotate(f'{y:.3f}', (x, y), textcoords='offset points',
                 xytext=(6, 8), fontsize=10, color='#c0392b')
ax1.axhline(B5, ls='--', color='#7f8c8d', lw=1.5,
            label=f'B5 全监督基线 0.397（5367 张）')
ax1.axhline(B6, ls='--', color='#2c3e50', lw=1.5,
            label=f'B6 全监督最优 0.416（5367 张，@1280）')
ax1.set_xlabel('已标注图像数', fontsize=11)
ax1.set_ylabel('test mask mAP@0.5', fontsize=11)
ax1.set_title('主动学习闭环：18% 标注预算的增长轨迹', fontsize=12)
ax1.legend(fontsize=9, loc='lower right')
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 0.48)

# ---- 右：每张标注的边际收益 ----
gain = [(mAP50[i] - mAP50[i-1]) / (n_labeled[i] - n_labeled[i-1])
        for i in range(1, len(n_labeled))]
labels = ['第 1 轮\n102→523\n(+421 张)', '第 2 轮\n523→944\n(+421 张)']
bars = ax2.bar(labels, gain, color=['#e67e22', '#2980b9'], width=0.5)
for b, g in zip(bars, gain):
    ax2.annotate(f'{g*10000:.1f} (x1e-4)/张', (b.get_x() + b.get_width()/2, g),
                 ha='center', va='bottom', fontsize=10)
ax2.set_ylabel('边际收益（ΔmAP50 / 新增标注）', fontsize=11)
ax2.set_title('每轮边际收益（random 策略，3 点尚无法判定递减趋势）', fontsize=12)
ax2.grid(axis='y', alpha=0.3)

fig.suptitle('AgroLabel-AL · 主动学习效果可视化（冒烟验证 2026-11-17，云端 96G 单卡）',
             fontsize=13, y=1.02)
fig.tight_layout()
out = 'docs/al_effect_smoke.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('saved:', out)
