# -*- coding: utf-8 -*-
"""生成 PlantSeg-34 Mask2Former 自包含配置（内联 pipeline，_delete_ 覆盖 dataset）。"""
import json
import mmengine
from mmengine.config import Config

REPO = '/root/autodl-tmp/Ryan/ArgoLable-AL'
OUT = REPO + '/configs_mmdet/mask2former/mask2former_r50_plantseg34_50e.py'

base = Config.fromfile(REPO + '/configs_mmdet/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py')

def pretty(obj):
    d = Config(dict(x=obj))
    txt = d.pretty_text
    return txt.split('x = ', 1)[1].rstrip()

cats = json.load(open(REPO + '/datasets/sam3/plantseg34_train.json'))['categories']
names = [c['name'] for c in sorted(cats, key=lambda c: c['id'])]
classes_str = ', '.join(repr(n) for n in names)
print('classes:', len(names), names[:3], names[-2:])

body = '''
_base_ = ['./mask2former_r50_8xb2-lsj-50e_coco.py']

num_things_classes = 34
num_stuff_classes = 0
num_classes = 34
metainfo = dict(classes=({classes_str},))

ANN_TRAIN = '/root/autodl-tmp/Ryan/ArgoLable-AL/datasets/sam3/plantseg34_train_mmdet.json'
ANN_VAL = '/root/autodl-tmp/Ryan/ArgoLable-AL/datasets/sam3/plantseg34_val_mmdet.json'
IMG_ROOT = 'data/plantseg34/'

train_dataloader = dict(
    batch_size=8, num_workers=8, persistent_workers=True,
    dataset=dict(_delete_=True,
        type='CocoDataset',
        metainfo=metainfo,
        data_root=IMG_ROOT,
        ann_file=ANN_TRAIN,
        data_prefix=dict(img_path='images/train/'),
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=True, min_size=8)))
val_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    dataset=dict(_delete_=True,
        type='CocoDataset',
        metainfo=metainfo,
        data_root=IMG_ROOT,
        ann_file=ANN_VAL,
        data_prefix=dict(img_path='images/val/'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(_delete_=True,
    type='CocoMetric', ann_file=ANN_VAL, metric=['bbox', 'segm'])
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=2,
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

content = ('train_pipeline = ' + pretty(base.train_pipeline) + '\n\n'
           + 'test_pipeline = ' + pretty(base.test_pipeline) + '\n\n'
           + body.replace('{classes_str}', classes_str))
with open(OUT, 'w') as f:
    f.write(content)
print('written:', OUT, len(content), 'bytes')

# 冒烟：加载配置验证可解析
c = Config.fromfile(OUT)
print('smoke: classes =', len(c.metainfo['classes']), '| ann =', c.train_dataloader.dataset.ann_file.split('/')[-1])
