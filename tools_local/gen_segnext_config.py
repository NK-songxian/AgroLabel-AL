# -*- coding: utf-8 -*-
"""生成 SegNeXt MSCAN-L PlantSeg-34 语义分割配置（40k iters，512 crop）。"""
import json

REPO = '/root/autodl-tmp/Ryan/ArgoLable-AL'
OUT = REPO + '/configs_mmseg/segnext/segnext_mscan-l_plantseg34-40k.py'

cats = json.load(open(REPO + '/datasets/sam3/plantseg34_train.json'))['categories']
names = [c['name'] for c in sorted(cats, key=lambda c: c['id'])]
classes_str = ',\n        '.join(repr(n) for n in names)

# 34 色调色板（仅可视化用）
palette = []
for i in range(len(names)):
    palette.append([(i * 47) % 256, (i * 97 + 60) % 256, (i * 151 + 120) % 256])
pal_str = ',\n        '.join(str(p) for p in palette)

cfg = f'''_base_ = './segnext_mscan-l_1xb16-adamw-160k_ade20k-512x512.py'

# PlantSeg-34：34 宿主类语义分割（0=背景，1-34=类，无 255 忽略区）
num_classes = 34
metainfo = dict(
    classes=(
        {classes_str},
    ),
    palette=(
        {pal_str},
    ))

data_root = 'data/plantseg34/'
img_suffix = '.jpg'
seg_map_suffix = '.png'

train_dataloader = dict(
    batch_size=8,
    num_workers=6,
    persistent_workers=True,
    dataset=dict(_delete_=True,
        type='CustomDataset',
        metainfo=metainfo,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='annotations_mmseg/train',
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        reduce_zero_label=False))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(_delete_=True,
        type='CustomDataset',
        metainfo=metainfo,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations_mmseg/val',
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        reduce_zero_label=False,
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# 160k -> 40k 迭代（数据集小 13 倍，步长等比例缩放）
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop',
                 max_iters=40000, val_interval=4000)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=40000,
         by_epoch=False),
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=4000, by_epoch=False,
                    max_keep_ckpts=2, save_best='mIoU', rule='greater'),
    logger=dict(type='LoggerHook', interval=50))

# crop 改 512 保持官方口径；随机翻转/缩放沿用 ade20k 管线
'''

with open(OUT, 'w') as f:
    f.write(cfg)
print('written:', OUT)

from mmengine.config import Config
c = Config.fromfile(OUT)
print('smoke OK | classes =', len(c.metainfo['classes']), '| iters =', c.train_cfg.max_iters)
