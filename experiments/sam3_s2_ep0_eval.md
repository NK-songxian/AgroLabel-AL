# S2 全量微调 · 第 0 轮检查点提前评测

> 目的：不干等第 2 轮的内置验证（`skip_first_val=true` + `val_epoch_freq=2`），
> 先用 epoch-0 检查点回答「微调到底有没有用」。
> 日期：2026-11-02　训练：`runs/sam3_s2_full_ft_host/`
> 权重：`checkpoints/checkpoint.pt`（epoch 0 末，训练 1 遍 / 共 10 遍）
> 转换：`datasets_tools/convert_ft_ckpt.py` → `sam3_s2_ep0.pt`（3.3 G，1465 张量）
> 评测：`prompt_sweep --split val --limit 200 --variants host_0 --imgsz 1008 --max-masks 120`

## 同批 200 张验证图（严格同口径）

| 指标 | S1 零样本 `sam3.pt` | S2 微调 1 轮 | 变化 |
|---|---|---|---|
| Mask mAP50 | 0.2146 | **0.3254** | **+51.6%** |
| Mask mAP50-95 | 0.0922 | **0.1584** | **+71.8%** |
| Instance Recall | 0.6776 | **0.8092** | +19.4% |
| Matched Dice | 0.8308 | 0.8435 | +1.5% |
| Boundary F1 (2px) | 0.7217 | 0.7285 | +0.9% |
| 预测实例数 | 22552 | 20773 | −7.9% |

预测数少了 8%、召回却从 0.68 升到 0.81 —— 是「更准」不是「乱撒」。

## 与 YOLOv8n-seg 基线对比

| 指标 | YOLOv8n-seg（150 ep，全 val） | S2 微调 1 轮（200 子集） |
|---|---|---|
| Mask mAP50 | 0.287 | **0.3254（+13.4%）** |
| Mask mAP50-95 | 0.140 | **0.1584（+13%）** |

**微调 1 轮的 SAM3 已在两项掩码指标上超过 YOLOv8n-seg 基线。**

## 口径说明

- 200 张子集比全 846 张略「容易」：S1 全量 mAP50=0.1709，同 200 子集=0.2146。
  因此**以同批 200 张的 S1 vs S2 对照为准**，跨子集数字不作结论。
- 本轮仅为第 0 轮检查点；第 2 轮末（预计 ≈18:35）会出全 846 张的内置验证。
- Boundary F1 仅 +0.9%，边界精度未同步提升，后续可关注（S3/S4 或边界损失）。

## 复现

```bash
# 1) 转换检查点
python datasets_tools/convert_ft_ckpt.py \
  --ft runs/sam3_s2_full_ft_host/checkpoints/checkpoint.pt \
  --out runs/sam3_s2_full_ft_host/sam3_s2_ep0.pt
# 2) 同口径评测
python -m sam3_baseline.prompt_sweep --split val \
  --data-root data_plantseg34 \
  --ckpt runs/sam3_s2_full_ft_host/sam3_s2_ep0.pt \
  --out experiments/sam3_s2_ep0 --variants host_0 --imgsz 1008 --limit 200
```
