# -*- coding: utf-8 -*-
"""D 线：Mask2Former-v2 PlantSeg-34 配置（常规 resize 配方）。

v1 复盘：mask2former_r50 8xb2-lsj-50e（512² 大尺度抖动裁剪）在 ep10 时 segm
mAP50 仅 0.118，被判定不敌 B5 终止。v2 假设：lsj 512 裁剪切断大图上下文 +
CropInst 单元过小，是 M2F 低分的主因，而非模型本身。
v2 改动：仅替换 pipeline 为常规 keep_ratio Resize(scale=(1280,640))（与 B6 的
有效分辨率对齐）+ RandomFlip，其余（34 类、50e、COCO 预训练、MultiStepLR）不变，
构成与 v1 的单变量对照。
"""
import json
import mmengine
from mmengine.config import Config

REPO = '/root/autodl-tmp/Ryan/ArgoLable-AL'
OUT = REPO + '/configs_mmdet/mask2former/mask2former_r50_plantseg34_50e_v2.py'

base = Config.fromfile(REPO + '/configs_mmdet/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py')

cats = json.load(open(REPO + '/datasets/sam3/plantseg34_train.json'))['categories']
names = [c['name'] for c in sorted(cats, key=lambda c: c['id'])]
classes_str = ', '.join(repr(n) for n in names)
print('classes:', len(names))

# 常规 resize 配方（v1 用 base 的 lsj pipeline，这里整体替换）
content = f'''
_base_ = ['./mask2former_r50_8xb2-lsj-50e_coco.py']

num_things_classes = 34
num_stuff_classes = 0
num_classes = 34
metainfo = dict(classes=({classes_str},))

ANN_TRAIN = '{REPO}/datasets/sam3/plantseg34_train_mmdet.json'
ANN_VAL = '{REPO}/datasets/sam3/plantseg34_val_mmdet.json'
IMG_ROOT = 'data/plantseg34/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # keep_ratio=False：统一变形 640x1280。keep_ratio=True 时 batch 内 mask 尺寸
    # 不一致会触发 mmdet DetDataPreprocessor pad 的 BitmapMasks reshape 崩溃
    # （lsj 配方靠固定 1024² 裁剪规避了该假设）。
    dict(type='Resize', scale=(1280, 640), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1280, 640), keep_ratio=False),
    dict(type='PackDetInputs')
]

# lsj base 强制 BatchFixedSizePad 到固定 1024x1024（配 lsj 裁剪用），
# 与本配方的 640x1280 统一尺寸冲突（BitmapMasks reshape 崩溃），清空之。
# 注意必须写 model.data_preprocessor——Runner 只读 model 内的这一份，
# base 顶层 data_preprocessor 变量与 model.data_preprocessor 是两份独立文本块。
data_preprocessor = dict(batch_augments=[])
model = dict(data_preprocessor=dict(batch_augments=[]))

train_dataloader = dict(
    batch_size=8, num_workers=8, persistent_workers=True,
    dataset=dict(_delete_=True,
        type='CocoDataset',
        metainfo=metainfo,
        data_root=IMG_ROOT,
        ann_file=ANN_TRAIN,
        data_prefix=dict(img=''),  # mmdet json 的 file_name 已含完整路径（mmdet 用键 'img' 而非 img_path）
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=True, min_size=8)))
val_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    dataset=dict(_delete_=True,
        type='CocoDataset',
        metainfo=metainfo,
        data_root=IMG_ROOT,
        ann_file=ANN_VAL,
        data_prefix=dict(img=''),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(_delete_=True,
    type='CocoMetric', ann_file=ANN_VAL, metric=['bbox', 'segm'])
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5, by_epoch=True,
                    max_keep_ckpts=2,
                    save_best='coco/segm_mAP', rule='greater'),
    logger=dict(type='LoggerHook', interval=50))
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop',
                 max_epochs=50, val_interval=5)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=50, by_epoch=True,
         milestones=[44, 49], gamma=0.1)
]

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_r50_8xb2-lsj-50e_coco/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth'
'''

with open(OUT, 'w') as f:
    f.write(content)
print('written:', OUT, len(content), 'bytes')

c = Config.fromfile(OUT)
tp = c.train_dataloader.dataset.pipeline
print('smoke: classes =', len(c.metainfo['classes']),
      '| pipeline[2] =', tp[2]['type'], tp[2]['scale'],
      '| keep_ratio =', tp[2]['keep_ratio'],
      '| epochs =', c.train_cfg.max_epochs)
