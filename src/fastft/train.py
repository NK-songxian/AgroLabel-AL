"""FastFT-SAM3 训练主脚本。

用法（云端仓库根目录）:
  PYTHONPATH=third_party/sam3_repo ./envs/sam3train/bin/python src/fastft/train.py \
      --exp fastft_i05 --size 504 --batch 16 --epochs 30 --patience 8

损失 = 类平衡 CE + 宏观 soft-Dice + 语义竞争 margin（GT 类须压过最强竞争类）。
验证 = 每 epoch 在 val 100 图上算 mIoU，按最优 mIoU 存 best.pt + 早停。
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import build_model, parameter_counts            # noqa: E402
from dataset import CocoSemantic, class_weights_from_coco  # noqa: E402

CKPT = str(ROOT / 'weights' / 'sam3.pt')


def dice_loss(logits, target, n_cls):
    probs = F.softmax(logits.float(), dim=1)[:, 1:]   # 丢弃背景通道，只算前景类
    onehot = F.one_hot(target, n_cls + 1).permute(0, 3, 1, 2).float()[:, 1:]
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    card = probs.sum(dims) + onehot.sum(dims)
    return (1 - (2 * inter / card.clamp_min(1))).mean()


def rival_margin_loss(logits, target, margin=0.3):
    """GT 像素上：GT 类 logit 须高出最强竞争类（含背景对手）margin。"""
    valid = target > 0
    if valid.sum() == 0:
        return logits.sum() * 0
    lg = logits.permute(0, 2, 3, 1)[valid]               # (N, C)
    tg = target[valid]                                   # (N,)
    gt = lg.gather(1, tg.unsqueeze(1)).squeeze(1)
    masked = lg.scatter(1, tg.unsqueeze(1), float('-inf'))  # 排除 GT 类本身
    rival = masked.max(1).values
    return F.relu(margin - (gt - rival)).mean()


@torch.no_grad()
def validate(model, loader, n_cls, device):
    model.eval()
    conf = torch.zeros(n_cls + 1, n_cls + 1, dtype=torch.int64, device=device)
    for batch in loader:
        img = batch['image'].to(device, non_blocking=True)
        tgt = batch['target'].to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(img)
        pred = out['logits'].float().argmax(1)
        idx = tgt * (n_cls + 1) + pred
        conf += torch.bincount(idx.flatten(), minlength=(n_cls + 1) ** 2).reshape(n_cls + 1, n_cls + 1)
    inter = conf.diag().float()
    union = conf.sum(0).float() + conf.sum(1).float() - inter
    iou = inter / union.clamp_min(1)
    model.train()
    return float(iou[1:].mean()), iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='fastft_i05')
    ap.add_argument('--size', type=int, default=504)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--lr-head', type=float, default=7e-4)
    ap.add_argument('--lr-lora', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--margin', type=float, default=0.3)
    ap.add_argument('--limit', type=int, default=0, help='>0 则只取前 N 张图（冒烟）')
    ap.add_argument('--lora-blocks', default='24,25,26,27,28,29,30,31')
    ap.add_argument('--lora-rank', type=int, default=8)
    a = ap.parse_args()

    torch.manual_seed(42)
    run_dir = ROOT / 'runtime' / 'runs' / a.exp
    run_dir.mkdir(parents=True, exist_ok=True)

    bank_path = ROOT / 'runtime' / 'assets' / 'fastft_text_bank.pt'
    if not bank_path.exists():
        raise SystemExit(f'缺文本库 {bank_path}，先跑 build_text_bank.py')

    tr = CocoSemantic(ROOT / 'datasets/sam3/plantseg34_train.json', ROOT / 'data/plantseg34',
                      size=a.size, train=True)
    va = CocoSemantic(ROOT / 'datasets/sam3/plantseg34_val.json', ROOT / 'data/plantseg34',
                      size=a.size, train=False)
    if a.limit:
        tr.ids = tr.ids[:a.limit]
        va.ids = va.ids[:min(a.limit, len(va.ids))]
    n_cls = tr.n_classes
    print(f'[data] train={len(tr)} val={len(va)} classes={n_cls}')

    wgt = class_weights_from_coco(ROOT / 'datasets/sam3/plantseg34_train.json', n_cls).to('cuda')

    model = build_model({'checkpoint': CKPT, 'text_bank': str(bank_path),
                         'input_size': a.size, 'head_width': 128, 'fpn_channels': 256,
                         'lora_blocks': [int(x) for x in a.lora_blocks.split(',')],
                         'lora_rank': a.lora_rank, 'lora_alpha': 2 * a.lora_rank})
    cnt = parameter_counts(model)
    print(f'[model] {cnt}')

    lora_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('vision.')]
    head_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('vision.')]
    opt = torch.optim.AdamW([{'params': lora_p, 'lr': a.lr_lora},
                             {'params': head_p, 'lr': a.lr_head}], weight_decay=0.01)
    base_lrs = [g['lr'] for g in opt.param_groups]
    steps_per_epoch = math.ceil(len(tr) / a.batch)
    total_steps = steps_per_epoch * a.epochs

    def lr_at(step):
        if step < a.warmup * steps_per_epoch:
            return step / max(1, a.warmup * steps_per_epoch)
        p = (step - a.warmup * steps_per_epoch) / max(1, total_steps - a.warmup * steps_per_epoch)
        return 0.5 * (1 + math.cos(math.pi * p))

    tl = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True,
                                     num_workers=a.workers, pin_memory=True, drop_last=True)
    vl = torch.utils.data.DataLoader(va, batch_size=a.batch, shuffle=False,
                                     num_workers=4, pin_memory=True)

    best, bad = -1.0, 0
    log_path = run_dir / 'train_log.jsonl'
    step = 0
    for ep in range(a.epochs):
        t0 = time.time()
        model.train()
        tot = 0.0
        for it, batch in enumerate(tl):
            img = batch['image'].to('cuda', non_blocking=True)
            tgt = batch['target'].to('cuda', non_blocking=True)
            lr_mul = lr_at(step)
            for g, base in zip(opt.param_groups, base_lrs):
                g['lr'] = base * lr_mul
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(img)
                logits = out['logits'].float()
                l_ce = F.cross_entropy(logits, tgt, weight=wgt)
                l_dc = dice_loss(logits, tgt, n_cls)
                l_rm = rival_margin_loss(logits, tgt, a.margin)
                loss = l_ce + 0.5 * l_dc + 0.3 * l_rm
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 1.0)
            opt.step()
            step += 1
            tot += float(loss)
            if it % 50 == 0:
                print(f'ep{ep} it{it}/{steps_per_epoch} loss={float(loss):.4f} '
                      f'(ce={float(l_ce):.3f} dice={float(l_dc):.3f} rival={float(l_rm):.3f}) '
                      f'lr_x{lr_mul:.2f} mem={torch.cuda.max_memory_allocated() / 2**30:.1f}G', flush=True)
        miou, iou = validate(model, vl, n_cls, 'cuda')
        rec = {'epoch': ep, 'train_loss': tot / max(1, it + 1), 'val_miou': miou,
               'sec': round(time.time() - t0, 1)}
        print(f'[epoch {ep}] loss={rec["train_loss"]:.4f} mIoU={miou:.4f} '
              f'{rec["sec"]}s bad={bad}', flush=True)
        with open(log_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        if miou > best:
            best, bad = miou, 0
            torch.save({'model': model.state_dict(), 'epoch': ep, 'miou': miou,
                        'cfg': vars(a)}, run_dir / 'best.pt')
            print(f'  -> best.pt (mIoU={miou:.4f})', flush=True)
        else:
            bad += 1
            if bad >= a.patience:
                print(f'[early-stop] {a.patience} epochs 无提升 (best mIoU={best:.4f})')
                break
    print(f'[done] best val mIoU = {best:.4f}')


if __name__ == '__main__':
    main()
