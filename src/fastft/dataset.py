"""COCO 实例标注 -> 逐像素 34 类标签（0=背景），letterbox 到固定方形。

- 最长边缩放到 size（不形变），补边到 size x size；补边区即背景。
- 训练增强：水平翻转 / 亮度对比度抖动 / 轻度尺度抖动。
- 只保留有标注的图（无标注图全背景，训练无意义）。
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as maskUtils


class CocoSemantic(torch.utils.data.Dataset):
    def __init__(self, coco_json, image_root, size=504, train=True):
        self.coco = json.load(open(coco_json))
        self.root = Path(image_root)
        self.size = size
        self.train = train
        self.imgs = {im['id']: im for im in self.coco['images']}
        self.anns = {}
        for a in self.coco['annotations']:
            self.anns.setdefault(a['image_id'], []).append(a)
        self.ids = [im['id'] for im in self.coco['images'] if im['id'] in self.anns]
        self.n_classes = len(self.coco['categories'])

    def __len__(self):
        return len(self.ids)

    def _rasterize(self, im):
        h, w = im['height'], im['width']
        lab = np.zeros((h, w), np.uint8)
        for a in self.anns[im['id']]:
            seg = a['segmentation']
            if isinstance(seg, list):
                rles = maskUtils.frPyObjects(seg, h, w)
                m = maskUtils.decode(maskUtils.merge(rles))
            else:
                m = maskUtils.decode(seg)
            if m.ndim == 3:
                m = m.max(axis=2)
            lab[m > 0] = a['category_id']
        return lab

    def __getitem__(self, i):
        im = self.imgs[self.ids[i]]
        img = cv2.imread(str(self.root / im['image_path']), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.root / im['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lab = self._rasterize(im)

        if self.train:
            if random.random() < 0.5:
                img, lab = img[:, ::-1], lab[:, ::-1]
            if random.random() < 0.8:
                alpha = 1.0 + random.uniform(-0.15, 0.15)
                beta = random.uniform(-12, 12)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        s = self.size / max(img.shape[:2])
        nw, nh = max(1, int(round(img.shape[1] * s))), max(1, int(round(img.shape[0] * s)))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        lab = cv2.resize(lab, (nw, nh), interpolation=cv2.INTER_NEAREST)

        canvas = np.zeros((self.size, self.size, 3), np.uint8)
        lcanvas = np.zeros((self.size, self.size), np.uint8)
        canvas[:nh, :nw] = img
        lcanvas[:nh, :nw] = lab

        rgb = torch.from_numpy(np.ascontiguousarray(canvas)).permute(2, 0, 1).float()
        rgb = rgb.div_(255).sub_(0.5).div_(0.5)  # [-1,1], CHW
        return {
            'image': rgb,
            'target': torch.from_numpy(lcanvas.astype(np.int64)),
            'image_id': im['id'],
        }


def class_weights_from_coco(coco_json, n_classes, mode='sqrt_inv'):
    """按标注面积估计类别频率 -> 损失类别权重（长尾平衡）。"""
    coco = json.load(open(coco_json))
    area = np.zeros(n_classes + 1, np.float64)
    for a in coco['annotations']:
        area[a['category_id']] += max(float(a.get('area', 0)), 1.0)
    area[area == 0] = area[area > 0].min()
    if mode == 'sqrt_inv':
        wgt = 1.0 / np.sqrt(area)
    else:
        wgt = 1.0 / area
    wgt = wgt / wgt.mean()
    wgt[0] = max(wgt.min(), 0.3)   # 背景压到较低权重
    return torch.tensor(wgt, dtype=torch.float32)
