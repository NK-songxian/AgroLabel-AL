# -*- coding: utf-8 -*-
"""把 plantseg34 COCO 多边形渲染成语义分割 png 掩码（mmseg 用，0=背景 1-34=宿主类）。"""
import json
import os
import numpy as np
from pycocotools.coco import COCO
from PIL import Image

REPO = '/root/autodl-tmp/Ryan/ArgoLable-AL'
DATA = REPO + '/data/plantseg34'

for split in ['train', 'val']:
    ann_file = f'{REPO}/datasets/sam3/plantseg34_{split}.json'
    coco = COCO(ann_file)
    out_dir = f'{DATA}/annotations_mmseg/{split}'
    os.makedirs(out_dir, exist_ok=True)
    n_done = 0
    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        # file_name 可能是 images/train/xxx.jpg 或 train2017/xxx.jpg
        rel = info['file_name']
        if rel.startswith('images/'):
            rel = rel[len('images/'):]
        base = os.path.splitext(os.path.basename(rel))[0]
        img_path = f'{DATA}/images/{rel}'
        H, W = info['height'], info['width']
        mask = np.zeros((H, W), dtype=np.uint8)  # 0=背景
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            m = coco.annToMask(ann).astype(bool)
            cid = ann['category_id']  # 1..34
            mask[m] = cid
        Image.fromarray(mask, mode='L').save(f'{out_dir}/{base}.png')
        n_done += 1
    print(split, 'rendered', n_done, 'masks ->', out_dir)
print('DONE')
