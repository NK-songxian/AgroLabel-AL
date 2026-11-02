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

---

## 内置验证（epoch 2 末，官方 34 类口径）

`val_epoch_freq=2` 首次触发于 2026-11-02 18:43，**150 张 val**（`limit_ids: 150`），
34 类**同时查询**、按类别匹配（`prompts` = 34 条 host 级短语）：

| 指标 | 值 |
|---|---|
| AP50-95 | 0.045 |
| AP50 | **0.084** |
| AP75 | 0.043 |
| AP_small / medium / large | 0.049 / 0.041 / 0.076 |
| AR@100 | 0.083 |
| 预测数 | 15000（100/图），score 中位 0.09 |

## ⚠️ 口径警告：0.325 与 0.084 不可直接比

| | 本文前面的 0.3254 | 内置验证 0.084 |
|---|---|---|
| 提示 | 每图**仅用自身 host**（oracle） | 34 类**全部**查询 |
| 类别判定 | **类别无关**（都算该 host） | **必须类别匹配** |
| 图像数 | 200 | 150 |

**结论修正：**
- **类别无关的病灶定位**：微调 1–2 轮可达 AP50 0.325（oracle host），明显优于零样本 0.215。
- **34 类条件检测（官方口径，与 YOLO 同口径）**：AP50 仅 **0.084，低于 YOLOv8n-seg 的 0.287**。
- 因此「微调 SAM3 已超 YOLO」**仅成立于 oracle-host 口径，不能作为 34 类任务的结论**，此前表述以此纠正。

**根因初判**：模型能定位病灶，但**宿主类别判别弱**；34 类同时查询产生大量跨类误检
（100 预/图、score 中位 0.09），precision 被拉低。

**下一步（对应手册 I-04 解耦）：**
1. 同 150 张上做「oracle-host vs 34-way」对照，量化类别惩罚；
2. 逐类混淆矩阵，确认是宿主混淆还是跨类重复；
3. 探索「类别无关病灶定位 + 独立宿主分类器」解耦方案。
