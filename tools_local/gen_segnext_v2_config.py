# -*- coding: utf-8 -*-
"""SegNeXt-v2 PlantSeg-34 配置：
1) decode_head.num_classes=35（0=bg + 1..34 类），修掉 v1 的 150 通道口径错误
2) CE 加 sqrt 逆频率类权重（治 Bean/Cauliflower/Zucchini 弱类），顺序 [bg, c1..c34]
3) 固定 Resize 512x512（多尺度 RandomResize 会触发 RandomCrop 小图 stack 崩溃，弃用）
4) 50k iters；从 v1 best 热启动（backbone/neck 加载，head conv_seg 形状不匹配自动跳过）
"""
import json
import numpy as np
from PIL import Image
import glob

from mmengine.config import Config

REPO = '/root/autodl-tmp/Ryan/ArgoLable-AL'
OUT = REPO + '/configs_mmseg/segnext/segnext_mscan-l_plantseg34-v2.py'

# ---- 类权重：train png 全量统计 sqrt 逆频率 ----
counts = np.zeros(35, dtype=np.int64)
for f in glob.glob(REPO + '/data/plantseg34/annotations_mmseg/train/*.png'):
    a = np.array(Image.open(f))
    counts += np.bincount(a.ravel(), minlength=35)
fg = counts[1:35].astype(np.float64)          # 34 个前景类
w = 1.0 / np.sqrt(np.maximum(fg, 1.0))
w = w / w.mean()                               # 均值归一
# 顺序必须与 png 标签一致：[bg=0.3, c1..c34]。此前把 bg 追加在末尾会整体错位一格。
# 注意长度必须是 35（F.cross_entropy 要求 == num_classes）；
# mmseg avg_factor 循环索引 ignore(255) 的 bug 由 site-packages 补丁另行修复。
full_w = [0.3] + w.tolist()
classes_str = ',\n        '.join(repr(n) for n in [c for c in json.load(
    open(REPO + '/datasets/sam3/plantseg34_train.json'))['categories']])
names = [c['name'] for c in sorted(json.load(
    open(REPO + '/datasets/sam3/plantseg34_train.json'))['categories'], key=lambda c: c['id'])]
w_str = ',\n        '.join(f'{x:.4f}' for x in full_w)
print('类像素数:', dict(zip(names, fg.astype(int))))
print('权重范围: min=%.3f max=%.3f' % (w.min(), w.max()))

palette = [(0, 0, 0)] + [[(i * 47) % 256, (i * 97 + 60) % 256, (i * 151 + 120) % 256]
                         for i in range(34)]
pal_str = ',\n        '.join(str(p) for p in palette)

cfg = f'''_base_ = './segnext_mscan-l_1xb16-adamw-160k_ade20k-512x512.py'

# v2：35 通道（0=bg + 34 宿主类）+ 类平衡 + 多尺度
num_classes = 35
metainfo = dict(
    classes=('__bg__',
        {', '.join(repr(n) for n in names)},
    ),
    palette=(
        {pal_str},
    ))

data_root = 'data/plantseg34/'
img_suffix = '.jpg'
seg_map_suffix = '.png'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    # 固定 512x512（v1 验证过的配方）。多尺度 RandomResize(640 短边) 在宽高比大的图上
    # 会缩出短边<512，mmseg RandomCrop 不补齐 -> batch stack 崩溃，故弃用。
    dict(scale=(512, 512), keep_ratio=False, type='Resize'),
    dict(cat_max_ratio=0.75, crop_size=(512, 512), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(2048, 640), type='Resize'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=6,
    persistent_workers=True,
    dataset=dict(_delete_=True,
        type='BaseSegDataset',
        metainfo=metainfo,
        data_root=data_root,
        data_prefix=dict(img_path='images/train', seg_map_path='annotations_mmseg/train'),
        pipeline=train_pipeline,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        reduce_zero_label=False))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(_delete_=True,
        type='BaseSegDataset',
        metainfo=metainfo,
        data_root=data_root,
        data_prefix=dict(img_path='images/val', seg_map_path='annotations_mmseg/val'),
        pipeline=test_pipeline,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        reduce_zero_label=False,
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], ignore_index=255)
test_evaluator = val_evaluator

model = dict(
    decode_head=dict(
        num_classes=35,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
            class_weight=({w_str},
            ))))  # [bg=0.3, c1..c34]：bg 低权重防淹没；须配合 site-packages 的 avg_factor 补丁

train_cfg = dict(_delete_=True, type='IterBasedTrainLoop',
                 max_iters=50000, val_interval=5000)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=50000,
         by_epoch=False),
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10000, by_epoch=False,
                    max_keep_ckpts=1, save_best='mIoU', rule='greater'),
    logger=dict(type='LoggerHook', interval=50))

# 从 v1 best 热启动：backbone/neck 加载，conv_seg 150->35 形状不匹配自动跳过（警告属预期）
load_from = '{REPO}/runtime/runs/segnext_l_plantseg34/best_mIoU_iter_40000.pth'
'''

with open(OUT, 'w') as f:
    f.write(cfg)
print('written:', OUT)

c = Config.fromfile(OUT)
head = c.model.decode_head
cw = head.loss_decode.class_weight
assert len(cw) == 35, f'class_weight 长度应为 35, 实际 {len(cw)}'
assert abs(cw[0] - 0.3) < 1e-6, 'bg 必须在首位'
print('smoke OK | head.num_classes =', head.num_classes,
      '| len(class_weight) =', len(cw),
      '| cw[0](bg) =', cw[0], '| cw[34](Zucchini) =', cw[34])
