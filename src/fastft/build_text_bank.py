"""离线构建 34 类文本概念库（移植 RivalScan cache_text.py 思想）。

把每个类别（宿主植物）的全部病征短语喂给官方冻结语言骨干，
掩码平均成单条短语向量，再对类内短语取均值 -> (34, 256) 概念库。
训练/推理时直接查表，语言侧计算 = 0。
只跑一次，产物落 runtime/assets/fastft_text_bank.pt。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco', default='datasets/sam3/plantseg34_train.json')
    ap.add_argument('--checkpoint', default='weights/sam3.pt')
    ap.add_argument('--out', default='runtime/assets/fastft_text_bank.pt')
    a = ap.parse_args()

    coco = json.load(open(ROOT / a.coco))
    cats = sorted(coco['categories'], key=lambda c: c['id'])
    phrases_per_cat = {c['id']: list(c['noun_phrases']) for c in cats}
    n = sum(len(v) for v in phrases_per_cat.values())
    print(f'[bank] {len(cats)} 类, 共 {n} 条病征短语')

    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(checkpoint_path=str(ROOT / a.checkpoint),
                                   load_from_HF=False, device='cuda',
                                   eval_mode=True, compile=False)
    language = model.backbone.language_backbone.eval()

    flat, spans = [], []
    for c in cats:
        start = len(flat)
        flat += phrases_per_cat[c['id']]
        spans.append((start, len(flat)))

    with torch.no_grad():
        padding, memory, _ = language(flat, device='cuda')
        w = (~padding).float().T[:, :, None]
        vec = (memory.float() * w).sum(0) / w.sum(0).clamp_min(1)   # (n_phrases, C)
        vectors = torch.stack([vec[s:e].mean(0) for s, e in spans])  # (34, C)
    print(f'[bank] 概念向量: {tuple(vectors.shape)}, 有限值: {bool(torch.isfinite(vectors).all())}')

    out = {
        'vectors': vectors.cpu(),
        'phrases': phrases_per_cat,
        'category_names': [c['name'] for c in cats],
        'category_ids': [c['id'] for c in cats],
        'checkpoint': a.checkpoint,
        'checkpoint_sha256': sha256_file(ROOT / a.checkpoint),
    }
    dst = ROOT / a.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    print(f'[bank] 已保存 {dst} (sha256={out["checkpoint_sha256"][:16]}...)')


if __name__ == '__main__':
    main()
