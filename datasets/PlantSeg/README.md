# datasets/PlantSeg — 数据接入说明

## 数据接入方式（重要）
主实验使用 **PlantSeg 34 类**（宿主级，仅植物物种，不含病害）。
数据**未复制到本仓库**，而是在 `configs/dataset.yaml` 中直接引用原始路径：

```
path: F:/_plantseg/dataset_plantseg_34
train: images/train
val:   images/val
test:  images/test
```

> 不复制的原因：原始数据约 7773 张图 + 标签，复制会浪费磁盘且与教程「第三步」的外部 `path` 设计冲突。
> 若后续需要本地副本，可把 `F:/_plantseg/dataset_plantseg_34/{images,labels}` 拷贝到本目录并改写 `dataset.yaml` 的 `path`。

## 原始数据结构（`F:/_plantseg/dataset_plantseg_34`）
```
dataset_plantseg_34/
├── images/{train,val,test}/     # 图片 (jpg)
├── labels/{train,val,test}/     # YOLO 实例分割标签 (txt，多边形归一化)
├── classes.json                 # 34 类定义 (mode=host)
└── data.yaml                    # nc=34, names
```
- 标签格式：`class_id x1 y1 x2 y2 ...`（多边形，归一化 0–1），由原始 PNG 掩码连通域重建。
- **本目录无原始 PNG 掩码**，仅有 mask 派生的分割多边形；若 BAGS/seg 模型需原始掩码须另补。

## 主动学习数据池（VARD-AL）
- `train` 全集 = 5367 张，拆分为：
  - **L₀（已标注初始集）** = `F:/_plantseg/_al/al34_random`（1443 张，占 train 26.89%）
  - **U₀（未标注候选池）** = train − L₀ = **3924** 张
- `val`(846) / `test`(1560) 在全部 AL 轮次中**固定不变**。
- ⚠️ **标签防泄漏**：U₀ 的标签虽在磁盘上存在，但 AL 选中前严禁读取；必须经 Oracle 接口「选中才给标签」。
- 详细统计见 `experiments/plantseg_34_audit.md` 与 `experiments/class_distribution.csv`。

## 类别覆盖（Task-002A 已确认）
34 类在 train/val/test **全部有有效样本**，无缺失。类别实例数 max/min 比约 36.8×，
存在不平衡，建议训练使用类别加权/平衡采样（重点：Basil、Lettuce、Squash 占比 <0.5%）。
