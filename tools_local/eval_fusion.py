# -*- coding: utf-8 -*-
"""C 线融合评测：检测器实例掩码 × SegNeXt 逐像素类别投票重定类。
对照组 A：检测器原始预测（同一 COCOeval 口径）
实验组 B：检测器掩码 + SegNeXt 多数投票定类
用法: python eval_fusion.py --det-w <best.pt> --seg-ck <v2 best> [--det-imgsz 1280]
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

from mmengine.config import Config
from mmseg.models import build_segmentor
from mmseg.utils import register_all_modules
register_all_modules(init_default_scope=True)

ROOT = Path('/root/autodl-tmp/Ryan/ArgoLable-AL')
ap = argparse.ArgumentParser()
ap.add_argument('--det-w', default='experiments/matrix/B5/runs/b5_y26x_960-2/weights/best.pt')
ap.add_argument('--det-imgsz', type=int, default=960)
ap.add_argument('--seg-cfg', default='configs_mmseg/segnext/segnext_mscan-l_plantseg34-40k.py')
ap.add_argument('--seg-ck', default='runtime/runs/segnext_l_plantseg34/best_mIoU_iter_40000.pth')
ap.add_argument('--out', default='runtime/runs/fusion_b5_segnext')
fa = ap.parse_args()
B5_W = ROOT / fa.det_w
SEG_CK = ROOT / fa.seg_ck
OUT_DIR = ROOT / fa.out
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- SegNeXt ----------
cfg = Config.fromfile(str(ROOT / fa.seg_cfg))
seg = build_segmentor(cfg.model)
ck = torch.load(str(SEG_CK), map_location='cpu', weights_only=False)
seg.load_state_dict(ck['state_dict'], strict=False)
seg = seg.cuda().eval()
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


@torch.no_grad()
def sem_labels(img_bgr, H, W):
    """SegNeXt 语义 argmax（0=bg,1..34），输出原图尺寸。"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    sc = min(2048.0 / W, 640.0 / H)
    nw, nh = max(1, int(round(W * sc))), max(1, int(round(H * sc)))
    r = cv2.resize(rgb, (nw, nh))
    ph, pw = (32 - nh % 32) % 32, (32 - nw % 32) % 32
    pad = np.zeros((nh + ph, nw + pw, 3), np.float32)
    pad[:nh, :nw] = (r - MEAN) / STD
    x = torch.from_numpy(pad.transpose(2, 0, 1))[None].cuda()
    lg = seg.decode_head(seg.extract_feat(x))[0, :35].float()
    lg = F.interpolate(lg[None], size=(H, W), mode='bilinear',
                       align_corners=False)[0].argmax(0).cpu().numpy()
    return lg


# ---------- B5 ----------
from ultralytics import YOLO
b5 = YOLO(str(B5_W))
print('[fusion] B5 + SegNeXt loaded', flush=True)

coco_gt = COCO(str(ROOT / 'datasets/sam3/plantseg34_val.json'))
val = json.load(open(ROOT / 'datasets/sam3/plantseg34_val.json'))
cats = sorted(val['categories'], key=lambda c: c['id'])
names = [c['name'] for c in cats]
NCLS = len(names)

preds_A, preds_B, votes_log = [], [], []
MIN_VOTE_RATIO = 0.30   # 掩码内非bg像素占比低于该值则保留 B5 原类
t0 = time.time()
n_changed = n_kept = 0
for i, im in enumerate(val['images']):
    H, W = im['height'], im['width']
    img = cv2.imread(str(ROOT / 'data/plantseg34' / im['image_path']))
    sem = sem_labels(img, H, W)
    r = b5.predict(img, imgsz=fa.det_imgsz, conf=0.001, iou=0.7, max_det=300, verbose=False)[0]
    if r.masks is None:
        continue
    masks = r.masks.data.cpu().numpy()          # (n, h/4?, w) — 已resize到imgsz栅格
    mh, mw = masks.shape[-2:]
    cls_ids = r.boxes.cls.cpu().numpy().astype(int) + 1   # ultralytics 0-based -> COCO 1-based
    scores = r.boxes.conf.cpu().numpy()
    # 掩码栅格 -> 原图
    if (mw, mh) != (W, H):
        masks = np.stack([cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR) for m in masks])
    masks = masks > 0.5
    for m, cid, sc_ in zip(masks, cls_ids, scores):
        area = int(m.sum())
        if area == 0:
            continue
        rle = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
        # A：B5 原始
        preds_A.append({'image_id': im['id'], 'category_id': int(cid),
                        'segmentation': {'counts': rle['counts'].decode('ascii'),
                                         'size': [H, W]}, 'score': float(sc_)})
        # B：投票重定类
        vals, cnts = np.unique(sem[m], return_counts=True)
        fg = [(v, c) for v, c in zip(vals, cnts) if v > 0]
        fg_ratio = sum(c for _, c in fg) / area
        if fg and fg_ratio >= MIN_VOTE_RATIO:
            new_cls = max(fg, key=lambda t: t[1])[0]
            preds_B.append({'image_id': im['id'], 'category_id': int(new_cls),
                            'segmentation': {'counts': rle['counts'].decode('ascii'),
                                             'size': [H, W]}, 'score': float(sc_)})
            votes_log.append((im['id'], int(cid), int(new_cls), round(fg_ratio, 3)))
            n_changed += int(new_cls != cid)
            n_kept += int(new_cls == cid)
        else:
            preds_B.append({'image_id': im['id'], 'category_id': int(cid),
                            'segmentation': {'counts': rle['counts'].decode('ascii'),
                                             'size': [H, W]}, 'score': float(sc_)})
            n_kept += 1
    if (i + 1) % 100 == 0:
        print(f'{i+1}/{len(val["images"])} {time.time()-t0:.0f}s', flush=True)


def run_eval(preds, tag):
    dt = coco_gt.loadRes(preds)
    E = COCOeval(coco_gt, dt, 'segm')
    E.evaluate(); E.accumulate(); E.summarize()
    print(f'[{tag}] mAP50={E.stats[1]:.4f} mAP50-95={E.stats[0]:.4f} mAP75={E.stats[2]:.4f}',
          flush=True)
    # per-class AP50
    p = E.eval['precision']            # [T,R,K,A,M]
    ap50 = p[0, :, :, 0, -1].mean(axis=1)
    weak = sorted(zip(names, ap50), key=lambda t: t[1])[:8]
    print(f'[{tag}] 弱类AP50:', ', '.join(f'{n}={v:.3f}' for n, v in weak), flush=True)
    return dict(mAP50=float(E.stats[1]), mAP50_95=float(E.stats[0]), mAP75=float(E.stats[2]),
                per_class_ap50={n: float(v) for n, v in zip(names, ap50)})


print(f'\n融合统计: 实例重定类 {n_changed} | 维持原类 {n_kept} | 总 {len(preds_B)}')
resA = run_eval(preds_A, 'A-B5原始')
resB = run_eval(preds_B, 'B-融合')
json.dump({'A': resA, 'B': resB, 'n_changed': n_changed,
           'min_vote_ratio': MIN_VOTE_RATIO},
          open(OUT_DIR / 'fusion_summary.json', 'w'), indent=2, default=float)
def _np_default(o):
    """numpy 标量/数组递归兜底（np.uint32 标量与 >1 维数组均会出现）。"""
    import numpy as _np
    if isinstance(o, _np.generic):
        return o.item()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    return str(o)


json.dump(preds_B, open(OUT_DIR / 'instances_val_fusion.json', 'w'), default=_np_default)
print('[fusion] DONE ->', OUT_DIR)
