# -*- coding: utf-8 -*-
"""Grounded-SAM-2 对照冒烟：Grounding DINO(HF, 免编译) 定位 + SAM2 掩码生成。

调研背景（2026-11-17）：调研报告建议以 Grounded-SAM-2 作为开放词汇定位对照。
我们已有的相邻证据：YOLO-World/YOLOE 零样本 box 0.004-0.007（Z2/Z3）；
SAM3 纯文本提示 0.0007-0.0102（Z1）；GT 框提示 SAM3 oracle 0.844（F4）。
本实验回答：Grounding DINO 定位 + SAM2 掩码在 PlantSeg 34 宿主类上能否超过
上述开放词汇基线。SAHI 切片仅 GDINO 1.5（付费 API）支持，本地路线不含。

用法（云端）：
  HF_ENDPOINT=https://hf-mirror.com python tools_local/gsam2_smoke.py --n 50
输出：runtime/runs/gsam2_smoke/summary.json + 同口径 COCOeval segm。
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path('/root/autodl-tmp/Ryan/ArgoLable-AL')
OUT_DIR = ROOT / 'runtime/runs/gsam2_smoke'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=50, help='val 前n图冒烟')
ap.add_argument('--conf', type=float, default=0.25)
ap.add_argument('--prompt-mode', default='host',
                choices=['host', 'lesion'],
                help='host=34 宿主短语列表映射 GT 类; lesion=病斑单类短语')
ap.add_argument('--device', default='cuda:0')
fa = ap.parse_args()

# ---------- Grounding DINO (HuggingFace, 无需编译 MSDA) ----------
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

GDINO_ID = 'IDEA-Research/grounding-dino-tiny'
processor = AutoProcessor.from_pretrained(GDINO_ID)
gdino = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to(fa.device).eval()

# ---------- SAM2 ----------
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2 = SAM2ImagePredictor.from_pretrained('facebook/sam2-hiera-small', device=fa.device)

cats = sorted(json.load(open(ROOT / 'datasets/sam3/plantseg34_val.json'))['categories'],
              key=lambda c: c['id'])
names = [c['name'] for c in cats]
assert len(names) == 34

# host 模式：34 类短语列表（Grounding DINO 以 ". " 分隔多短语，输出 label 文本回映射）
host_prompt = ' . '.join(names) + ' .'
lesion_prompt = 'leaf lesion . disease spot on leaf . plant disease . leaf spot .'
prompt = host_prompt if fa.prompt_mode == 'host' else lesion_prompt
name2cid = {n: c['id'] for n, c in zip(names, cats)}

val_json = json.load(open(ROOT / 'datasets/sam3/plantseg34_val.json'))
images = val_json['images'][:fa.n]
coco_gt = COCO(str(ROOT / 'datasets/sam3/plantseg34_val.json'))

results = []
t0 = time.time()
n_box = n_mask = 0
for i, im in enumerate(images):
    H, W = im['height'], im['width']
    img = cv2.cvtColor(cv2.imread(str(ROOT / 'data/plantseg34' / im['image_path'])),
                       cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        inputs = processor(images=img, text=prompt, return_tensors='pt').to(fa.device)
        outs = gdino(**inputs)
        proc = processor.post_process_grounded_object_detection(
            outs, input_sizes=inputs['input_sizes'] if hasattr(inputs, 'input_sizes') else None,
            threshold=fa.conf, text_threshold=0.25)[0]
    boxes = proc['boxes'].cpu().numpy()
    phrases = [str(t).lower().strip(' .') for t in proc['text_labels']
               if True] if 'text_labels' in proc else [str(t).lower() for t in proc['labels']]
    n_box += len(boxes)
    if len(boxes) == 0:
        continue
    sam2.set_image(img)
    with torch.no_grad():
        masks, scores, _ = sam2.predict(box=boxes, multimask_output=False)
    masks = masks[:, 0] > 0            # (n, H, W) bool
    for m, ph, sc in zip(masks, phrases, scores):
        if m.sum() < 64:
            continue
        n_mask += 1
        cid = name2cid.get(ph, 1)      # 未匹配短语回退 Apple(1)，lesion 模式全记 1
        rle = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
        results.append({'image_id': im['id'], 'category_id': int(cid),
                        'segmentation': {'counts': rle['counts'].decode('ascii'),
                                         'size': [H, W]},
                        'score': float(sc)})
    if (i + 1) % 10 == 0:
        print(f'{i+1}/{len(images)} box={n_box} mask={n_mask} '
              f'{time.time()-t0:.0f}s', flush=True)

print(f'\n[gsam2-{fa.prompt_mode}] {len(images)} 图: 框 {n_box}, 掩码 {n_mask}, '
      f'耗时 {time.time()-t0:.0f}s')
summary = {'n_images': len(images), 'prompt_mode': fa.prompt_mode,
           'prompt': prompt, 'conf': fa.conf, 'n_boxes': n_box,
           'n_masks': n_mask, 'elapsed_s': round(time.time() - t0, 1)}
if results:
    dt = coco_gt.loadRes(results)
    E = COCOeval(coco_gt, dt, 'segm')
    E.params.imgIds = [im['id'] for im in images]
    E.evaluate(); E.accumulate(); E.summarize()
    summary['segm_mAP50'] = float(E.stats[1])
    summary['segm_mAP50_95'] = float(E.stats[0])
    print(f'[gsam2] mask mAP50={E.stats[1]:.4f} mAP50-95={E.stats[0]:.4f}')

json.dump(summary, open(OUT_DIR / f'summary_{fa.prompt_mode}.json', 'w'),
          indent=2, default=float)
json.dump(results, open(OUT_DIR / f'instances_{fa.prompt_mode}.json', 'w'),
          default=float)
print('[gsam2] DONE ->', OUT_DIR)
