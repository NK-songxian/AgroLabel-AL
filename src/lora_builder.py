# -*- coding: utf-8 -*-
"""
lora_builder.py —— 给 SAM3 官方训练管线用的 LoRA 版模型构造器（矩阵 I-01 / I-02）

SAM3 官方 `sam3/train/train.py` 通过 Hydra 的 `_target_` 直接构造模型：

    model:
      _target_: sam3.model_builder.build_sam3_image_model
      bpe_path: ...
      checkpoint_path: ...
      enable_segmentation: true

本文件提供**同签名 + 额外 LoRA 参数**的替代品，只做三件事：
  1. 先按官方方式构造并加载预训练 SAM3（保证能读到权重）；
  2. 再把 nn.Linear 就地换成 LoRALinear（src/lora_inject.py）；
  3. 冻结基座，只放开 lora_* 与 `lora_extra_trainable` 指定的模块。

生成配置时把 `_target_` 换成 `src.lora_builder.build_sam3_lora` 即可，
其余训练/数据/优化器配置全部沿用 F1，保证与全参微调**同口径可比**——
这是 I-01 消融能站得住的前提：差的只有"改哪些参数"，不是"换了套管线"。

运行要求：`PYTHONPATH` 里要有仓库根（`training/run_sam3_ft.py` 会自动设），
因为 Hydra 在 fork 出来的进程里也要能 import 到本模块。
"""
from __future__ import annotations

from typing import Any

try:  # 正常路径：仓库根在 sys.path
    from src.lora_inject import inject_lora, mark_only_lora_trainable, param_stats
except Exception:  # noqa: BLE001  兜底：src 本身在 sys.path
    from lora_inject import inject_lora, mark_only_lora_trainable, param_stats  # type: ignore


def build_sam3_lora(bpe_path: str | None = None,
                    checkpoint_path: str | None = None,
                    load_from_HF: bool = False,
                    device: str = "cpus",
                    eval_mode: bool = False,
                    enable_segmentation: bool = True,
                    model_builder_module: str = "sam3.model_builder",
                    model_builder_fn: str = "build_sam3_image_model",
                    lora_rank: int = 8,
                    lora_alpha: float = 16.0,
                    lora_dropout: float = 0.0,
                    lora_targets: str = "",
                    lora_extra_trainable: str = "",
                    lora_verbose: bool = True,
                    **kwargs: Any):
    """构造 SAM3 并注入 LoRA。参数与 `build_sam3_image_model` 兼容，多出 lora_* 一组。"""
    import importlib

    mod = importlib.import_module(model_builder_module)
    build = getattr(mod, model_builder_fn)

    kw = dict(bpe_path=bpe_path, checkpoint_path=checkpoint_path,
              load_from_HF=load_from_HF, device=device,
              eval_mode=eval_mode, enable_segmentation=enable_segmentation)
    kw.update(kwargs)
    kw = {k: v for k, v in kw.items() if v is not None}

    model = build(**kw)
    if lora_verbose:
        print("[LoRA] 基座已加载（先加载权重、后注入，避免 state_dict 名字错位）")

    targets = tuple(t.strip() for t in (lora_targets or "").split(",") if t.strip())
    extra = tuple(t.strip() for t in (lora_extra_trainable or "").split(",") if t.strip())

    st = inject_lora(model, rank=lora_rank, alpha=lora_alpha,
                     dropout=lora_dropout, targets=targets, verbose=lora_verbose)
    st2 = mark_only_lora_trainable(model, extra=extra)
    if lora_verbose:
        print(f"[LoRA] 冻结基座后：可训练 {st2['trainable']:,} / {st2['total']:,} "
              f"= {st2['ratio'] * 100:.3f}%（extra={extra}）")
        if st2["trainable"] == 0:
            raise SystemExit("[LoRA] 可训练参数为 0 —— 检查 lora_targets / extra 配置")
    model._lora_stats = st2
    model._lora_cfg = {"rank": lora_rank, "alpha": lora_alpha,
                       "dropout": lora_dropout, "targets": targets, "extra": extra}
    return model


def probe(model) -> dict:
    """调试用：返回 LoRA 统计。"""
    return {"stats": param_stats(model),
            "cfg": getattr(model, "_lora_cfg", {})}
