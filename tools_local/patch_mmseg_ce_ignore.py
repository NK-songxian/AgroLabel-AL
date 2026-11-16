# -*- coding: utf-8 -*-
"""mmseg 1.2.2 cross_entropy_loss.py 的 ignore-index 越界补丁（幂等）。

背景：decode_head.loss_by_feat 对非空 class_weight 走 avg_factor 分支，
mmseg 原实现逐像素 `class_weight[cls] for cls in label`。当标签含
ignore_index=255（RandomCrop pad 小图 / seg_pad_val）且权重表长度只有
num_classes=35 时，CUDA IndexKernel 越界 device-side assert。

修复：忽略像素权重置 0，仅对有效像素索引权重表。
幂等：已打过补丁（源码含 PATCH(plantseg) 标记）则跳过。

用法（云端）：
    python tools_local/patch_mmseg_ce_ignore.py
"""
import sys

TARGET = ('/root/miniconda3/lib/python3.12/site-packages/'
          'mmseg/models/losses/cross_entropy_loss.py')

OLD = '''        else:
            # the average factor should take the class weights into account
            label_weights = torch.stack([class_weight[cls] for cls in label
                                         ]).to(device=class_weight.device)

            if avg_non_ignore:
                label_weights[label == ignore_index] = 0
            avg_factor = label_weights.sum()'''

NEW = '''        else:
            # the average factor should take the class weights into account
            # PATCH(plantseg): ignore_index(255) 也会被 class_weight[label] 索引，
            # 35 长度权重表 + 255 pad 标签会越界 -> 忽略像素权重置 0
            valid_mask = label != ignore_index
            label_weights = torch.zeros(
                label.shape, dtype=class_weight.dtype,
                device=class_weight.device)
            label_weights[valid_mask] = class_weight[
                label[valid_mask].long()]

            avg_factor = label_weights.sum()'''

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else TARGET
    src = open(path).read()
    if 'PATCH(plantseg)' in src:
        print('already patched, skip')
        sys.exit(0)
    assert OLD in src, '原代码块未找到，mmseg 版本可能不匹配'
    open(path, 'w').write(src.replace(OLD, NEW))
    print('patched OK:', path)
