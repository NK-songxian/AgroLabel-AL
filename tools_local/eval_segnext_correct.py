# -*- coding: utf-8 -*-
"""SegNeXt 正确口径评测：argmax 限前 35 通道(0=bg,1..34=类)。
输出: ① per-class IoU（含背景参考）② 连通域实例 -> COCOeval segm（fastft 口径 conf=0.5 min_area=64）。
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image

from mmengine.config import Config
from mmseg.models import build_segmentor
from mmseg.utils import register_all_modules
register_all_modules(init_default_scope=True)

ROOT = Path('/root/autodl-tmp/Ryan/ArgoLable-AL')

ap = argparse.ArgumentParser()
ap.add_argument('--conf', type=float, default=0.5)
ap.add_argument('--min-area', type=int, default=64)
a = ap.parse_args()

cfg = Config.fromfile(str(ROOT / 'configs_mmseg/segnext/segnext_mscan-l_plantseg34-40k.py'))
model = build_segmentor(cfg.model)
ck = torch.load(str(ROOT / 'runtime/runs/segnext_l_plantseg34/best_mIoU_iter_40000.pth'),
                map_location='cpu', weights_only=False)
model.load_state_dict(ck['state_dict'], strict=False)
model = model.cuda().eval()

cats = json.load(open(ROOT / 'datasets/sam3/plantseg34_val.json'))['categories']
names = [c['name'] for c in sorted(cats, key=lambda c: c['id'])]
assert len(names) == 34

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

val_json = json.load(open(ROOT / 'datasets/sam3/plantseg34_val.json'))
anns_by_img = {}
for an in val_json['annotations']:
    anns_by_img.setdefault(an['image_id'], []).append(an)

coco_gt = COCO(str(ROOT / 'datasets/sam3/plantseg34_val.json'))
results = []
n_cls = 35
conf_mat = np.zeros((n_cls, n_cls), dtype=np.int64)  # [gt, pred]

t0 = time.time()
for i, im in enumerate(val_json['images']):
    H, W = im['height'], im['width']
    img = cv2.imread(str(ROOT / 'data/plantseg34' / im['image_path']))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # 测试口径与训练 val 一致：keep_ratio 缩放至 fit (2048,512)
    sc = min(2048.0 / W, 512.0 / H)
    nw, nh = max(1, int(round(W * sc))), max(1, int(round(H * sc)))
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    img_n = (img_r - MEAN) / STD
    ph = (32 - nh % 32) % 32
    pw = (32 - nw % 32) % 32
    img_p = np.zeros((nh + ph, nw + pw, 3), dtype=np.float32)
    img_p[:nh, :nw] = img_n
    x = torch.from_numpy(img_p.transpose(2, 0, 1))[None].cuda()

    meta = dict(ori_shape=(H, W), img_shape=(nh, nw),
                pad_shape=(nh + ph, nw + pw), scale_factor=(sc, sc),
                flip=False, img_path='')
    with torch.no_grad():
        feat = model.extract_feat(x)
        seg_logits = model.decode_head(feat)  # (1,150,h',w')
    logits = seg_logits[0, :n_cls].float().cpu().numpy()
    # 手动 resize 回原图（双线性），与 GT 像素级对齐
    logits = cv2.resize(logits.transpose(1, 2, 0), (W, H),
                        interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
    probs = F.softmax(torch.from_numpy(logits), dim=0).numpy()
    pred = logits.argmax(0)

    # GT 语义 png（0=bg,1..34）
    gt = np.array(Image.open(str(ROOT / 'data/plantseg34/annotations_mmseg/val'
                                / (Path(im['image_path']).stem + '.png'))))
    idx = gt.astype(np.int64).ravel() * n_cls + pred.astype(np.int64).ravel()
    conf_mat += np.bincount(idx, minlength=n_cls * n_cls).reshape(n_cls, n_cls)

    # 实例化（fastft 口径）：c=1..34 -> category_id=c 与 COCO GT 直接对齐
    for c in range(1, 35):
        m = (probs[c] > a.conf).astype(np.uint8)
        if m.sum() == 0:
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        for k in range(1, n):
            x0, y0, w0, h0, area = stats[k]
            if area < a.min_area:
                continue
            comp = (lab == k)
            score = float(probs[c][comp].mean())
            rle = maskUtils.encode(np.asfortranarray(comp.astype(np.uint8)))
            results.append({'image_id': im['id'], 'category_id': c,
                            'segmentation': {'counts': rle['counts'].decode('ascii'),
                                             'size': [H, W]},
                            'score': score})
    if (i + 1) % 100 == 0:
        print(f'{i+1}/{len(val_json["images"])} inst={len(results)} {time.time()-t0:.0f}s', flush=True)

# ---------- per-class IoU（不含背景） ----------
iou_rows = []
for c in range(1, 35):
    inter = conf_mat[c, c]
    union = conf_mat[c, :].sum() + conf_mat[:, c].sum() - inter
    iou = inter / union if union else float('nan')
    iou_rows.append((names[c - 1], float(iou)))
ious = np.array([r[1] for r in iou_rows], dtype=np.float64)
miou = float(np.nanmean(ious))

bg_inter = conf_mat[0, 0]
bg_union = conf_mat[0, :].sum() + conf_mat[:, 0].sum() - bg_inter
print('\n===== per-class IoU (bg 除外, 34 类) =====')
for name, v in sorted(iou_rows, key=lambda r: r[1]):
    print(f'{name:20s} {v:.4f}')
print(f'背景 IoU = {bg_inter/bg_union:.4f}')
print(f'== mIoU(34 类) = {miou:.4f} ==')

# ---------- COCOeval segm ----------
run_dir = ROOT / 'runtime/runs/segnext_l_plantseg34'
json.dump(results, open(run_dir / 'instances_val.json', 'w'))
coco_dt = coco_gt.loadRes(results)
E = COCOeval(coco_gt, coco_dt, 'segm')
E.evaluate(); E.accumulate(); E.summarize()
summary = {'mIoU_34class': miou, 'bg_IoU': float(bg_inter / bg_union),
           'mAP50_95': float(E.stats[0]), 'mAP50': float(E.stats[1]),
           'mAP75': float(E.stats[2]), 'n_instances': len(results),
           'conf': a.conf, 'min_area': a.min_area,
           'note': 'argmax 限 35 通道(0=bg)；per-class IoU 含 Zucchini；实例=连通域 fastft 口径'}
json.dump(summary, open(run_dir / 'eval_summary_correct.json', 'w'), indent=2)
print('[eval] mask mAP50-95=%.4f | mAP50=%.4f | mAP75=%.4f | 实例 %d'
      % (E.stats[0], E.stats[1], E.stats[2], len(results)))
