"""FastFT-SAM3 评测：语义 logits -> 连通域实例 -> COCOeval mask AP50-95。

与项目矩阵同一口径（pycocotools COCOeval segm）。用法:
  PYTHONPATH=third_party/sam3_repo ./envs/sam3train/bin/python src/fastft/predict_eval.py \
      --run fastft_i05 [--conf 0.5] [--limit 0]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import build_model            # noqa: E402
from dataset import CocoSemantic         # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='fastft_i05')
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--min-area', type=int, default=64)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    run_dir = ROOT / 'runtime' / 'runs' / a.run
    ckpt = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']
    print(f"[eval] {a.run}: epoch={ckpt['epoch']} val_mIoU={ckpt['miou']:.4f} conf={a.conf}")

    bank = torch.load(ROOT / 'runtime' / 'assets' / 'fastft_text_bank.pt',
                      map_location='cpu', weights_only=False)
    model = build_model({'checkpoint': str(ROOT / 'weights' / 'sam3.pt'),
                         'text_bank': str(ROOT / 'runtime' / 'assets' / 'fastft_text_bank.pt'),
                         'input_size': cfg['size'], 'head_width': 128, 'fpn_channels': 256,
                         'lora_blocks': [int(x) for x in cfg['lora_blocks'].split(',')],
                         'lora_rank': cfg['lora_rank'], 'lora_alpha': 2 * cfg['lora_rank']})
    model.load_state_dict(ckpt['model'])
    model.eval().cuda()

    va = CocoSemantic(ROOT / 'datasets/sam3/plantseg34_val.json', ROOT / 'data/plantseg34',
                      size=cfg['size'], train=False)
    if a.limit:
        va.ids = va.ids[:a.limit]
    n_cls = va.n_classes
    coco_gt = COCO(str(ROOT / 'datasets/sam3/plantseg34_val.json'))

    results = []
    t0 = time.time()
    for i in range(len(va)):
        item = va[i]
        img = item['image'][None].cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(img)['logits'].float()
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()          # (C+1, S, S)
        im = va.imgs[item['image_id']]
        H, W = im['height'], im['width']
        # 数据集是 letterbox：内容在画布左上 (nh, nw)，必须先裁掉补边区再 resize 回原图，
        # 否则非正方形图的预测整体被压偏、与 GT 空间错位。
        sc = cfg['size'] / max(H, W)
        nw, nh = max(1, int(round(W * sc))), max(1, int(round(H * sc)))
        probs = probs[:, :nh, :nw]
        probs = cv2.resize(probs.transpose(1, 2, 0), (W, H),
                           interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        for c in range(1, n_cls + 1):
            m = (probs[c] > a.conf).astype(np.uint8)
            if m.sum() == 0:
                continue
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            for k in range(1, n):
                x, y, w, h, area = stats[k]
                if area < a.min_area:
                    continue
                comp = (lab == k)
                score = float(probs[c][comp].mean())
                rle = maskUtils.encode(np.asfortranarray(comp.astype(np.uint8)))
                results.append({'image_id': im['id'], 'category_id': c,
                                'segmentation': {'counts': rle['counts'].decode('ascii'),
                                                 'size': [H, W]},
                                'score': score})
        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{len(va)} inst={len(results)} {time.time()-t0:.0f}s', flush=True)

    print(f'[eval] 共 {len(results)} 实例, 耗时 {time.time()-t0:.0f}s')
    if not results:
        print('[eval] 无预测，跳过 COCOeval')
        return
    dt = json.load(open(ROOT / 'datasets/sam3/plantseg34_val.json'))
    res_path = run_dir / 'instances_val.json'
    json.dump(results, open(res_path, 'w'))
    coco_dt = coco_gt.loadRes(results)
    E = COCOeval(coco_gt, coco_dt, 'segm')
    E.evaluate(); E.accumulate(); E.summarize()
    summary = {'mAP50_95': float(E.stats[0]), 'mAP50': float(E.stats[1]),
               'mAP75': float(E.stats[2]), 'n_instances': len(results),
               'conf': a.conf, 'min_area': a.min_area, 'val_miou_ckpt': ckpt['miou']}
    json.dump(summary, open(run_dir / 'eval_summary.json', 'w'), indent=2)
    print('[eval] mask mAP50-95 = %.4f | mAP50 = %.4f | mAP75 = %.4f'
          % (summary['mAP50_95'], summary['mAP50'], summary['mAP75']))


if __name__ == '__main__':
    main()
