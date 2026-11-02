"""把 SAM3 训练产出的 checkpoint 转成 Ultralytics / 原版 `sam3.pt` 兼容格式。

为什么需要
----------
`sam3/train/trainer.py` 存的是 `{"model": <Sam3Image.state_dict()>, ...}`，
键是不带前缀的（`backbone.vision_backbone...`、`segmentation_head...`）。
而官方发布权重 `sam3.pt` 是**扁平 dict**，键带 `detector.` / `tracker.` 前缀：

    {'detector.backbone.vision_backbone.trunk.pos_embed': ..., 'tracker.no_mem_embed': ...}

推理侧（Ultralytics `SAM3SemanticPredictor` 或 `model_builder._load_checkpoint`）
都按这个前缀解析，所以要把训练权重套回 `detector.` 命名，`tracker.*` 原样保留。

用法:
    python datasets_tools/convert_ft_ckpt.py \
        --ft runs/sam3_s2_full_ft_host/checkpoints/checkpoint.pt \
        --base /root/autodl-tmp/Ryan/sam3.pt \
        --out weights/sam3_plantseg34_s2.pt
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", required=True, help="训练产出的 checkpoint.pt")
    ap.add_argument("--base", default="/root/autodl-tmp/Ryan/sam3.pt", help="原始发布权重")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=True)
    ft = torch.load(args.ft, map_location="cpu", weights_only=True)
    sd = ft["model"] if isinstance(ft, dict) and "model" in ft else ft

    # 训练权重是全量微调后的 Sam3Image.state_dict()，键不带前缀
    if any(k.startswith("detector.") for k in sd):
        raise SystemExit("[FAIL] 输入权重已带 detector. 前缀，无需转换")
    if any(k.startswith("tracker.") for k in sd):
        raise SystemExit("[FAIL] 输入权重含 tracker. 键，实际结构与本脚本预期不符")

    base_det = {k for k in base if k.startswith("detector.")}
    ft_det = {"detector." + k for k in sd}

    missing = sorted(base_det - ft_det)
    extra = sorted(ft_det - base_det)
    if missing or extra:
        print(f"[WARN] 与发布权重不完全对齐：base 独有 {len(missing)}，训练独有 {len(extra)}")
        for k in missing[:5]:
            print("   base-only:", k)
        for k in extra[:5]:
            print("   ft-only  :", k)
    else:
        print(f"[OK] detector 键完全对齐：{len(base_det)} 个")

    # 保留 base 的全部键（含训练时未构建的 sam2_convs.*，属实例交互分支，
    # 与本次微调无关，原样保留才能让推理侧按原结构 strict 加载），
    # 再用训练权重覆盖所有 detector.*。
    out = dict(base)
    out.update({"detector." + k: v for k, v in sd.items()})
    torch.save(out, args.out)
    print(f"[OK] 已写出 {args.out}（{len(out)} 个张量）")
    if missing:
        print(f"     其中 {len(missing)} 个 detector 键沿用 base 原值（训练未构建该分支）")


if __name__ == "__main__":
    main()
