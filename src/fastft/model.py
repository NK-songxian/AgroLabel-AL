"""FastFT-SAM3：冻结官方视觉主干 + 末端 QV-LoRA + 文本缓存语义竞争头。

移植 RivalScan-SAM3 三项核心技术，适配 34 类植物病害逐像素多类分割：
  1) 冻结 MLP 走 no_grad 捷径（enable_frozen_mlp_autograd）——冻结前缀完全不建 autograd 图；
  2) QV-LoRA 经 parametrize 注入官方联合 QKV 权重（B 零初始化，注入不改变初始输出）；
  3) 语义竞争：文本概念 query 与逐像素特征 cosine -> 类间 logit，
     概念相似度图以固定权重直通最终分割 logits（正类须压过最强竞争类）。
语言骨干完全不上 GPU 参与训练（文本库离线预编码）。
"""
import gc
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize


# ---------------------------------------------------------------- 官方视觉 PEFT

class QVLowRank(nn.Module):
    """只对官方联合 QKV 权重注入 Q/V 低秩增量（K 保持不动）。"""

    def __init__(self, width, rank=8, alpha=16):
        super().__init__()
        self.Aq = nn.Parameter(torch.empty(rank, width))
        self.Av = nn.Parameter(torch.empty(rank, width))
        self.Bq = nn.Parameter(torch.zeros(width, rank))
        self.Bv = nn.Parameter(torch.zeros(width, rank))
        nn.init.kaiming_uniform_(self.Aq, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Av, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, weight):
        width = self.Aq.shape[1]
        if weight.shape != (3 * width, width):
            raise ValueError('期望官方联合 QKV 权重 (3W,W)，实际 ' + str(tuple(weight.shape)))
        q = self.Bq @ self.Aq
        v = self.Bv @ self.Av
        return weight + self.scale * torch.cat((q, torch.zeros_like(q), v), dim=0)


def enable_frozen_mlp_autograd():
    """冻结的官方 MLP 在 no_grad 下直算——冻结前缀完全不建 autograd 图（速度核心）。"""
    from sam3.model.vitdet import Mlp
    if getattr(Mlp, '_fastft_autograd', False):
        return
    original = Mlp.forward

    def forward(self, x):
        if not torch.is_grad_enabled():
            return original(self, x)
        if not x.requires_grad and not any(p.requires_grad for p in self.parameters()):
            with torch.no_grad():
                return original(self, x)
        x = self.act(self.fc1(x))
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return self.drop2(x)

    Mlp.forward = forward
    Mlp._fastft_autograd = True


# ---------------------------------------------------------------- 轻量多类头

def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                         nn.GroupNorm(min(8, o), o), nn.SiLU())


class LocalBlock(nn.Module):
    def __init__(self, a, b, stride=1):
        super().__init__()
        self.skip = nn.Conv2d(a, b, 1, stride, bias=False) if a != b or stride != 1 else nn.Identity()
        self.path = nn.Sequential(
            nn.Conv2d(a, b, 3, stride, 1, bias=False), nn.GroupNorm(8, b), nn.SiLU(),
            nn.Conv2d(b, b, 3, 1, 1, groups=b, bias=False), nn.GroupNorm(8, b), nn.SiLU(),
            nn.Conv2d(b, b, 1, bias=False))

    def forward(self, x):
        return F.silu(self.skip(x) + self.path(x))


def resize(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class SemanticCompeteHead(nn.Module):
    """CNN 多分辨率分支 + FPN 融合 + 34 类文本概念竞争 + 高分辨率细化。"""

    def __init__(self, text_vectors, width=128, fpn_channels=256, n_levels=4):
        super().__init__()
        n_cls = text_vectors.shape[0]
        self.n_cls = n_cls
        self.register_buffer('text_vectors', F.normalize(text_vectors.float(), dim=1))
        self.full = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1), nn.SiLU())
        self.local_half = LocalBlock(3, 32, 2)
        self.quarter = nn.Sequential(LocalBlock(32, 64, 2), LocalBlock(64, 64))
        self.eighth = nn.Sequential(LocalBlock(64, 96, 2), LocalBlock(96, 128))
        self.global_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(fpn_channels, 64, 1), nn.GroupNorm(8, 64), nn.SiLU())
            for _ in range(n_levels)])
        self.level_weights = nn.Parameter(torch.zeros(n_levels))
        self.fuse = nn.Sequential(block(64 + 64 + 128, width), LocalBlock(width, width))
        self.text_proj = nn.Sequential(nn.Linear(text_vectors.shape[1], width), nn.LayerNorm(width))
        self.query_offset = nn.Parameter(torch.randn(n_cls, width) * .01)
        self.query_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.visual_proj = nn.Conv2d(width, width, 1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(5.)))
        self.refine = nn.Sequential(block(width + 32, 64), LocalBlock(64, 64))
        self.detail = nn.Sequential(block(64 + 16, 32), LocalBlock(32, 32))
        self.segmentation = nn.Conv2d(32, n_cls, 1)
        # 显式背景通道：概念竞争的"第 0 个对手"，像素须赢过背景才算前景
        self.bg_head = nn.Sequential(block(width, 32), nn.Conv2d(32, 1, 1))

    def semantics(self, x):
        """文本概念与逐像素特征的竞争相似度 (B, n_cls, h, w)。"""
        b, c, h, w = x.shape
        query = self.text_proj(self.text_vectors) + self.query_offset      # (K, width)
        query = query[None].expand(b, -1, -1)
        context = F.adaptive_avg_pool2d(x, (8, 8)).flatten(2).transpose(1, 2)
        delta, _ = self.query_attention(query, context, context, need_weights=False)
        query = F.normalize((query + delta).float(), dim=-1)
        visual = F.normalize(self.visual_proj(x).float(), dim=1)
        return torch.einsum('bkc,bchw->bkhw', query, visual) * self.logit_scale.float().clamp(0, 3).exp()

    def forward(self, features, rgb):
        half = self.local_half(rgb)
        quarter = self.quarter(half)
        eighth = self.eighth(quarter)
        maps = [resize(p(v), quarter.shape[-2:]) for p, v in zip(self.global_proj, features)]
        w = self.level_weights.softmax(0)
        glob = sum(a * b for a, b in zip(w, maps))
        x = self.fuse(torch.cat([glob, quarter, resize(eighth, quarter.shape[-2:])], 1))
        sem_q = self.semantics(x)                                   # (B,K,qh,qw)
        sem_full = resize(sem_q, rgb.shape[-2:])
        refined = self.refine(torch.cat([resize(x, half.shape[-2:]), half], 1))
        refined = self.detail(torch.cat([resize(refined, rgb.shape[-2:]), self.full(rgb)], 1))
        cls_logits = self.segmentation(refined) + 0.25 * sem_full   # 语义竞争余量直通
        bg = resize(self.bg_head(x), rgb.shape[-2:])                # (B,1,H,W) 背景对手
        logits = torch.cat([bg, cls_logits], dim=1)                 # (B, n_cls+1, H, W)
        return {'logits': logits, 'aux_semantic': sem_full, 'concept_logits': sem_q}


# ---------------------------------------------------------------- 整机

class FastFTModel(nn.Module):
    def __init__(self, vision, head, input_size):
        super().__init__()
        self.vision = vision
        self.head = head
        self.input_size = input_size

    def train(self, mode=True):
        super().train(mode)
        # 冻结主干永远 eval：关 dropout/随机深度；LoRA 梯度不受影响。
        self.vision.eval()
        return self

    def forward(self, image):
        """image: RGB float, 已归一化到 [-1,1]，任意分辨率（保持原纵横比）。"""
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('期望 B,3,H,W RGB')
        native = F.interpolate(image, (self.input_size, self.input_size),
                               mode='bilinear', align_corners=False)
        features, _, _, _ = self.vision(native)
        return self.head(features, image)


def build_model(cfg, device='cuda'):
    cfg = dict(cfg)
    enable_frozen_mlp_autograd()
    from sam3.model_builder import build_sam3_image_model
    source = build_sam3_image_model(checkpoint_path=cfg['checkpoint'], load_from_HF=False,
                                    device=device, eval_mode=True, compile=False)
    vision = source.backbone.vision_backbone
    for p in vision.parameters():
        p.requires_grad_(False)
    blocks = vision.trunk.blocks
    selected = cfg['lora_blocks']
    if not selected or min(selected) < 0 or max(selected) >= len(blocks):
        raise ValueError('非法 LoRA block 索引')
    for index in selected:
        layer = blocks[index].attn.qkv
        adapter = QVLowRank(layer.in_features, cfg['lora_rank'], cfg['lora_alpha']).to(layer.weight.device)
        parametrize.register_parametrization(layer, 'weight', adapter)
    resolution = cfg['input_size']
    if resolution % 14:
        raise ValueError('SAM3 patch14 要求分辨率能被 14 整除')
    grid = resolution // 14
    for layer in blocks:
        if layer.window_size == 0 and tuple(layer.attn.input_size) != (grid, grid):
            layer.attn.input_size = (grid, grid)
            layer.attn._setup_rope_freqs()
            layer.attn.to(device)
    vision.trunk.use_act_checkpoint = False
    bank = torch.load(cfg['text_bank'], map_location='cpu', weights_only=False)
    head = SemanticCompeteHead(bank['vectors'], width=cfg['head_width'],
                               fpn_channels=cfg['fpn_channels'],
                               n_levels=len(vision.convs)).to(device)
    del source
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return FastFTModel(vision, head, resolution).to(device).train()


def parameter_counts(model):
    return {'total': sum(p.numel() for p in model.parameters()),
            'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'lora': sum(p.numel() for n, p in model.named_parameters()
                        if p.requires_grad and n.startswith('vision.')),
            'head': sum(p.numel() for p in model.head.parameters() if p.requires_grad)}
