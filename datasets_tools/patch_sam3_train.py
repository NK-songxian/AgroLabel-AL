"""给 sam3 官方仓库打最小训练兼容补丁（幂等）。

背景
----
`sam3/model/vitdet.py` 里 ViT trunk 的 `Mlp.forward` 直接调用
`sam3/perflib/fused.addmm_act`，而该融合算子内部有

    if torch.is_grad_enabled():
        raise ValueError("Expected grad to be disabled.")

也就是说这份 release 的 trunk MLP 是**只支持推理**的。做全量微调（S2）时
grad 必然是开着的，第一步 forward 就会崩。这里把前向改成：有梯度走原始
fc1+act，无梯度仍走融合算子（不改变推理性能、不改变数学等价性）。

用法:
    python datasets_tools/patch_sam3_train.py --repo /root/autodl-tmp/Ryan/sam3_repo

幂等：已打过补丁会直接跳过。
"""

from __future__ import annotations

import argparse
from pathlib import Path

ORIG_B = """    common_params["matcher"] = matcher
    model = Sam3Image(**common_params)

    return model
"""

PATCHED_B = """    common_params["matcher"] = matcher
    model = Sam3Image(**common_params)

    # [AgroLabel-AL patch] segmentation_head.semantic_seg_head 只在启用
    # `loss_fn_semantic_seg` 时才会产生梯度。我们的微调只用实例 mask loss，
    # 若不冻结，DDP 会因"参数未参与损失"在第二迭代直接报错。
    if not eval_mode and segmentation_head is not None:
        seg_sem = getattr(segmentation_head, "semantic_seg_head", None)
        if seg_sem is not None:
            seg_sem.requires_grad_(False)

    return model
"""

ORIG = """    def forward(self, x):
        x = addmm_act(type(self.act), self.fc1, x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
"""

PATCHED = """    def forward(self, x):
        # [AgroLabel-AL patch] 原实现直接走 fused.addmm_act，该算子要求 grad 关闭，
        # 因此无法训练。有梯度时退回等价的 fc1+act，无梯度时保留融合路径。
        if torch.is_grad_enabled():
            x = self.act(self.fc1(x))
        else:
            x = addmm_act(type(self.act), self.fc1, x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/root/autodl-tmp/Ryan/sam3_repo")
    args = ap.parse_args()

    # --- patch 1: 训练时 trunk MLP 走非融合路径 ---
    target = Path(args.repo) / "sam3" / "model" / "vitdet.py"
    text = target.read_text(encoding="utf-8")
    if "[AgroLabel-AL patch]" in text:
        print(f"[SKIP] 已打过补丁: {target}")
    elif ORIG not in text:
        raise SystemExit(f"[FAIL] 没找到待替换的 Mlp.forward 原文，请人工检查 {target}")
    else:
        target.write_text(text.replace(ORIG, PATCHED, 1), encoding="utf-8")
        print(f"[OK] 已打补丁: {target}")

    # --- patch 2: 冻结 semantic_seg_head（避免 DDP 未参与损失报错）---
    target_b = Path(args.repo) / "sam3" / "model_builder.py"
    text_b = target_b.read_text(encoding="utf-8")
    if "seg_sem.requires_grad_(False)" in text_b:
        print(f"[SKIP] 已打过补丁: {target_b}")
    elif ORIG_B not in text_b:
        raise SystemExit(f"[FAIL] 没找到待替换的 _create_sam3_model 结尾，请人工检查 {target_b}")
    else:
        target_b.write_text(text_b.replace(ORIG_B, PATCHED_B, 1), encoding="utf-8")
        print(f"[OK] 已打补丁: {target_b}")


if __name__ == "__main__":
    main()
