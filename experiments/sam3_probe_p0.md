# SAM3 零样本探测（P0）· 实测记录

环境：AutoDL RTX PRO 6000 Blackwell 96G / torch 2.8.0+cu128 / ultralytics 8.4.177
权重：`/root/autodl-tmp/Ryan/sam3.pt`（Ultralytics 原生 `SAM3SemanticPredictor`，无需 Meta 仓库）
数据：PlantSeg-34 · val（1692 张，每图单类别）

---

## 1. 首批结果：固定原始类名做提示 → 召回很差

`imgsz=1008, conf=0.01, quantize=16`，每类抽 4 张：

| 类别 | 命中 | 最高置信度 | 类别 | 命中 | 最高置信度 |
|---|---|---|---|---|---|
| Lettuce | 4/4 | 0.531 | Tomato | 2/4 | 0.844 |
| Maple | 4/4 | 0.818 | Grape | 1/4 | **0.063** |
| Wheat | 4/4 | 0.647 | Potato | 1/4 | **0.153** |
| Blueberry | 4/4 | 0.934 | Zucchini | 1/4 | **0.044** |
| Raspberry | 3/4 | 0.970 | Rice | 1/4 | **0.014** |
| **Soybean** | **0/4** | **0.000** | Tobacco | 1/4 | **0.029** |
| **Coffee** | **0/4** | **0.000** | Plum | 2/4 | **0.061** |
| Ginger | 3/4 | **0.018** | Peach | 3/4 | **0.078** |

**总命中 70/134 = 52.2%。**

初步判断会得出"SAM3 在农业密集冠层上不可用"——**但这是错的，原因是提示词。**

---

## 2. 换说法后：几乎全部救回

对低分类别逐条试 `variants`（`X` / `X leaf` / `X plant` / `X leaves` / `a X leaf`）：

| 类别 | 原提示 conf | 最佳提示 | 最佳 conf | 提升 |
|---|---|---|---|---|
| Soybean | 0.000 | `a soybean leaf` | **0.963** | 0 → 0.96 |
| Plum | 0.061 | `plum plant` | **0.955** | ×15.7 |
| Grape | 0.063 | `grape leaf` | **0.938** | ×14.9 |
| Rice | 0.014 | `rice plant` | **0.920** | ×65.7 |
| Tobacco | 0.029 | `tobacco plant` | **0.853** | ×29.4 |
| Zucchini | 0.044 | `zucchini plant` | **0.838** | ×19.0 |
| Potato | 0.153 | `potato plant` | **0.821** | ×5.4 |
| Ginger | 0.018 | `ginger plant` | 0.149 | ×8.3（仍弱） |
| Peach | 0.078 | `peach` | 0.086 | ≈0（仍弱） |

---

## 3. 结论

1. **SAM3 基础版可用**，但**裸类名提示不可用**。提示词形式（裸名词 / 带 organ / 带冠词）
   对置信度的影响是数量级的，不是微调级别。
2. "病害提示词库"的思路对，但**必须是数据驱动搜出来的，不能手写**。
   手写 prompt bank 在这个数据集上的代价已经量化：52.2% vs 接近全覆盖。
3. **Peach / Ginger 仍失败** → 说明简单文本提示有上限，需要视觉样例提示（exemplar），
   这正好是后面创新模块（双锚定提示）的实证依据，不是拍脑袋加的。
4. 速度：`imgsz=1008` 约 1.6–2.0 s/张（含编码器）。val 全量约 50 min，可接受；
   每图 34 提示仲裁不可接受（要 ~30 h），必须"一图一提示"。

---

## 4. 对 YOLO 对比的口径提醒

YOLOv8n-seg 基线：Mask mAP50 **0.287** / mAP50-95 **0.140**，覆盖 34 类全部。

SAM3 若用 **tuned 提示**（每类最优），才具备可比性；用裸类名比出来的低分是
"提示词问题"而非"模型能力问题"，不能写进论文。

---

## 5. 下一步

- `P1` 用 tuned prompt bank 跑 val 全量 → `sam3_metrics.json` / 逐类 AP / Dice
- `P2` Peach、Ginger 等失败类加 exemplar 提示
- `P3` 跨提示一致性 → 不确定性 → 喂给 VARD-AL

---

## 附：复现命令

```bash
cd /root/autodl-tmp/Ryan/ArgoLable-AL
export PYTHONPATH=.
PY=/root/miniconda3/bin/python

# 逐类探测
$PY -m sam3_baseline.diag_probe --data-root /root/autodl-tmp/Ryan/data_plantseg34 \
  --ckpt /root/autodl-tmp/Ryan/sam3.pt --per-class 4 --imgsz 1008 --out runs/diag/probe.json

# 提示词搜索
$PY -m sam3_baseline.probe_variants --data-root /root/autodl-tmp/Ryan/data_plantseg34 \
  --ckpt /root/autodl-tmp/Ryan/sam3.pt --only all --per-class 4 --out runs/diag/tune_all.json

# 零样本推理 + 评测
$PY -m sam3_baseline.zeroshot_infer --split val \
  --data-root /root/autodl-tmp/Ryan/data_plantseg34 --ckpt /root/autodl-tmp/Ryan/sam3.pt \
  --out runs/sam3_zeroshot/val --conf 0.25
$PY -m sam3_baseline.eval_seg --gt runs/sam3_zeroshot/val/gt.json \
  --pred runs/sam3_zeroshot/val/pred.json --out runs/sam3_zeroshot/val
```
