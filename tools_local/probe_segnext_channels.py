# -*- coding: utf-8 -*-
"""单图探针：验证 SegNeXt 通道-标签映射。"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mmengine.config import Config
from mmseg.models import build_segmentor
from mmseg.utils import register_all_modules
register_all_modules(init_default_scope=True)

ROOT = Path('/root/autodl-tmp/Ryan/ArgoLable-AL')
cfg = Config.fromfile(str(ROOT / 'configs_mmseg/segnext/segnext_mscan-l_plantseg34-40k.py'))
model = build_segmentor(cfg.model)
ck = torch.load(str(ROOT / 'runtime/runs/segnext_l_plantseg34/best_mIoU_iter_40000.pth'),
                map_location='cpu', weights_only=False)
model.load_state_dict(ck['state_dict'], strict=False)
model = model.cuda().eval()

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

val = json.load(open(ROOT / 'datasets/sam3/plantseg34_val.json'))
# 找一张 apple 开头与一张 zucchini 开头的图
targets = {}
for im in val['images']:
    b = Path(im['image_path']).stem.lower()
    for key in ('apple_scab', 'zucchini_powdery'):
        if b.startswith(key) and key not in targets:
            targets[key] = im

for key, im in targets.items():
    H, W = im['height'], im['width']
    img = cv2.imread(str(ROOT / 'data/plantseg34' / im['image_path']))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    sc = min(2048.0 / W, 512.0 / H)
    nw, nh = int(round(W * sc)), int(round(H * sc))
    img_r = cv2.resize(img, (nw, nh))
    img_p = np.zeros(((nh + 31) // 32 * 32, (nw + 31) // 32 * 32, 3), np.float32)
    img_p[:nh, :nw] = (img_r - MEAN) / STD
    x = torch.from_numpy(img_p.transpose(2, 0, 1))[None].cuda()
    with torch.no_grad():
        feat = model.extract_feat(x)
        lg = model.decode_head(feat)[0, :35].float()  # (35,h',w')
    gt = np.array(Image.open(str(ROOT / 'data/plantseg34/annotations_mmseg/val'
                                / (Path(im['image_path']).stem + '.png'))))
    print(f'\n== {key}: {W}x{H} | logits spatial = {tuple(lg.shape[1:])} | GT unique = '
          f'{sorted(np.unique(gt).tolist())}')
    lg_up = F.interpolate(lg[None], size=(H, W), mode='bilinear', align_corners=False)[0]
    probs = F.softmax(lg_up, dim=0).cpu().numpy()
    for lab in sorted(np.unique(gt).tolist()):
        m = gt == lab
        sel = probs.reshape(35, -1)[:, m.ravel()]  # (35, npx)
        top = np.argsort(-sel.mean(1))[:4]
        print(f'  GT={lab:2d} (npx={m.sum():7d}) -> 通道概率Top4: '
              + ', '.join(f'ch{k}={sel[k].mean():.3f}' for k in top))
