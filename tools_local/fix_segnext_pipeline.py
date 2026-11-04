# -*- coding: utf-8 -*-
"""给 segnext plantseg34 配置补回被 _delete_ 清掉的 pipeline。"""
from mmengine.config import Config

REPO = '/root/autodl-tmp/Ryan/ArgoLable-AL'
P = REPO + '/configs_mmseg/segnext/segnext_mscan-l_plantseg34-40k.py'

base = Config.fromfile(REPO + '/configs_mmseg/segnext/segnext_mscan-l_1xb16-adamw-160k_ade20k-512x512.py')

def pretty(obj):
    d = Config(dict(x=obj))
    return d.pretty_text.split('x = ', 1)[1].rstrip()

s = open(P).read()
s = ('train_pipeline = ' + pretty(base.train_pipeline) + '\n\n'
     + 'test_pipeline = ' + pretty(base.test_pipeline) + '\n\n'
     + s)
# dataloader 里接回 pipeline
s = s.replace("seg_map_path='annotations_mmseg/train'),\n        img_suffix=",
              "seg_map_path='annotations_mmseg/train'),\n        pipeline=train_pipeline,\n        img_suffix=")
s = s.replace("seg_map_path='annotations_mmseg/val'),\n        img_suffix=",
              "seg_map_path='annotations_mmseg/val'),\n        pipeline=test_pipeline,\n        img_suffix=")
open(P, 'w').write(s)

c = Config.fromfile(P)
ds = c.train_dataloader.dataset
print('smoke OK | pipeline len =', len(ds['pipeline']), '| type =', ds['type'])
