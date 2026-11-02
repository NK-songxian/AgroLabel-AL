# -*- coding: utf-8 -*-
"""
lora_inject.py —— 通用 LoRA 注入器（矩阵 I-01 / I-02 的地基）

不依赖任何第三方 PEFT 库，只用 torch，把模型里的 `nn.Linear` 就地替换成

    LoRALinear:  y = W x + (alpha / r) * B A x

其中 `lora_B` 初始化为 0，所以**注入当下模型输出与原来逐位相同**——这一点很关键：
注入后可以直接继承原始权重跑一次验证，确认没被改坏，再开始训练。

## 两条必须遵守的约定

1) **先加载权重，后注入。** 注入会把参数名从 `xxx.weight` 变成
   `xxx.base.weight`（外加 `xxx.lora_A/lora_B`），先注入再 `load_state_dict`
   会大面积 key 不匹配。

2) **只训练 lora 参数。** 注入后调用 `mark_only_lora_trainable(model, extra)`，
   基座全部 `requires_grad=False`，显存里省掉绝大部分梯度和优化器状态——
   96G 卡跑 SAM3 全参微调吃紧，LoRA 是让它"能跑第二组、第三组消融"的前提。

## 命名

`xxx` 保留原路径，所以在 SAM3 官方训练配置里，
`param_names: ['backbone.vision_backbone.*']` 这类按名字分组的 LR 规则仍然成立。

## 自检

    python src/lora_inject.py --self-test

会用一个小玩具模型验证：注入后输出不变、lora_B 非零后输出按公式变化、
只有 lora 参数可训练。这个自检纯 CPU、秒级，可以在任何机器上跑。
"""
from __future__ import annotations

import argparse
import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """把 nn.Linear 包成 y = Wx + scaling * B A x（B 零初始化）。"""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank 必须为正，得到 {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        dt = out.dtype
        # A/B 用 float32 存（小矩阵，稳），x 可能是 bf16，做一次对齐避免 dtype 冲突
        z = self.dropout(x).to(self.lora_A.dtype) @ self.lora_A.t()
        z = z @ self.lora_B.t()
        return out + (z.to(dt) * self.scaling)

    def extra_repr(self) -> str:
        return (f"in={self.base.in_features}, out={self.base.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}")


def _parent_and_leaf(model: nn.Module, full_name: str):
    parts = full_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora(model: nn.Module,
                rank: int = 8,
                alpha: float = 16.0,
                dropout: float = 0.0,
                targets: Sequence[str] = (),
                exclude: Iterable[str] = ("lora_A", "lora_B"),
                max_modules: int = 0,
                verbose: bool = True) -> dict:
    """把名字里含 `targets` 任一子串的 nn.Linear 换成 LoRALinear。

    targets 为空表示"全部 Linear"。返回统计信息（注入数、可训练参数等）。
    """
    excl = tuple(exclude or ())
    hits = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or isinstance(mod, LoRALinear):
            continue
        if any(e in name for e in excl):
            continue
        if targets and not any(t in name for t in targets):
            continue
        hits.append(name)
    if max_modules:
        hits = hits[:max_modules]
    if not hits:
        raise SystemExit(
            "没有匹配到任何 nn.Linear —— 检查 --lora-targets 子串是否正确。\n"
            "提示：先打印模型的模块名再挑目标，见 src/lora_inject.py --list-modules <ckpt>")

    for name in hits:
        parent, leaf = _parent_and_leaf(model, name)
        setattr(parent, leaf, LoRALinear(getattr(parent, leaf), rank, alpha, dropout))

    st = param_stats(model)
    st.update({"rank": rank, "alpha": alpha, "dropout": dropout,
               "n_injected": len(hits), "injected": hits[:20]})
    if verbose:
        print(f"[LoRA] 注入 {len(hits)} 个 Linear（rank={rank}, alpha={alpha}, "
              f"dropout={dropout}）")
        print(f"[LoRA] 可训练参数 {st['trainable']:,} / {st['total']:,} "
              f"= {st['ratio'] * 100:.3f}%")
        print(f"[LoRA] 前几个：{hits[:5]}")
    return st


def param_stats(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    return {"total": int(total), "trainable": int(train), "lora": int(lora),
            "ratio": (train / total) if total else 0.0}


def mark_only_lora_trainable(model: nn.Module, extra: Sequence[str] = ()) -> dict:
    """冻结全部，只放开 lora_* 与名字含 `extra` 子串的参数。"""
    for n, p in model.named_parameters():
        keep = ("lora_" in n) or any(e and e in n for e in extra)
        p.requires_grad = keep
    return param_stats(model)


def lora_state_dict(model: nn.Module) -> dict:
    """只导出 lora 参数（体积通常 <10MB，方便回传/复现）。"""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def merge_lora(model: nn.Module) -> int:
    """把 LoRA 权重加进基座并还原成 nn.Linear（推理加速用）。返回合并数。"""
    n = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, LoRALinear):
            continue
        with torch.no_grad():
            mod.base.weight.data += (mod.lora_B @ mod.lora_A) * mod.scaling
        parent, leaf = _parent_and_leaf(model, name)
        setattr(parent, leaf, mod.base)
        n += 1
    return n


def list_linear_modules(model: nn.Module, head: int = 40) -> None:
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    print(f"共 {len(names)} 个 nn.Linear，前 {min(head, len(names))} 个：")
    for n in names[:head]:
        print("  ", n)


# ---------------------------------------------------------------- self test
def _self_test() -> None:
    torch.manual_seed(0)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.blocks = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])
            self.head = nn.Linear(16, 4)

        def forward(self, x):
            for b in self.transformer.blocks:
                x = torch.relu(b(x))
            return self.head(x)

    m = Toy().eval()
    x = torch.randn(3, 16)
    y0 = m(x).clone()

    st = inject_lora(m, rank=4, alpha=8.0, targets=("transformer",))
    assert st["n_injected"] == 2, st

    # 1) B 为 0 时注入不改变输出
    y1 = m(x)
    assert torch.allclose(y0, y1, atol=1e-6), (y0 - y1).abs().max()
    print("[self-test] 注入后输出与原模型一致 ✓")

    # 2) 手动置 B 后，输出必须等于 base(x) + scaling * (x A^T B^T)
    blk = m.transformer.blocks[0]
    with torch.no_grad():
        blk.lora_B.normal_()
    xin = torch.relu(x)  # blocks[0] 的输入就是 x
    manual = blk.base(xin) + (xin @ blk.lora_A.t() @ blk.lora_B.t()) * blk.scaling
    got = blk(xin)
    assert torch.allclose(manual, got, atol=1e-5), (manual - got).abs().max()
    print("[self-test] LoRA 前向公式一致 ✓")

    # 3) 冻结后只有 lora 参数可训练
    st2 = mark_only_lora_trainable(m)
    assert st2["trainable"] == st2["lora"] > 0, st2
    print(f"[self-test] 仅 {st2['trainable']} 个参数可训练（占 {st2['ratio']*100:.2f}%）✓")

    # 4) merge 后与 LoRA 前向等价
    import copy

    m4 = copy.deepcopy(m)
    y_before = m4(x).clone()
    n = merge_lora(m4)
    y_after = m4(x)
    assert n == 2 and torch.allclose(y_before, y_after, atol=1e-5), (y_before - y_after).abs().max()
    print("[self-test] merge_lora 与 LoRA 前向等价 ✓")
    print("[self-test] 全部通过")


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA 注入器 / 自检")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--list-modules", default="", help="打印某模型（.pt）里的 Linear 名字")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--targets", default="", help="逗号分隔的模块名子串")
    a = ap.parse_args()

    if a.self_test:
        _self_test()
        return
    if a.list_modules:
        # SAM3 的 .pt 需要 build 之后再 load，这里只做"能读就列"的粗查
        import ultralytics  # noqa: F401

        from ultralytics.models.sam import SAM3SemanticPredictor

        p = SAM3SemanticPredictor(overrides=dict(task="segment", mode="predict",
                                                 model=a.list_modules, verbose=False))
        p.setup_model()
        list_linear_modules(p.model if not hasattr(p.model, "model") else p.model)
        return
    ap.error("要么 --self-test，要么 --list-modules <ckpt>")


if __name__ == "__main__":
    main()
