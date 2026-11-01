# PlantSeg 34 类 · 实验协议（Task-003B）

> 本文记录 YOLOv8n-seg 基线训练的数据协议。所有数字来自真实文件统计
> （Task-002A 审计：`experiments/plantseg_34_audit.md` / `class_distribution.csv`）。

## 1. 主数据集选择理由（为何用 34 类而非 115 类）

- **任务定位**：本原型目标是“多模态数据智能预标注 + 主动学习”闭环，基线需要**稳定、可解释、类别不过细**的病害分割模型。
- **115 类是“植物+病害”细粒度组合**，长尾更严重，且 val/test 各缺 1 类（`Peach___peach rust`、`Coffee___coffee black rot`），
  细类评估不稳定，不利于 AL 效率对比。
- **34 类是宿主级（植物物种）**，val/test 全覆盖、每类样本量更充足，适合作为主动学习对照的“干净基线”。
- 因此主实验采用 **34 类**；115 类保留为后续细粒度扩展实验。

## 2. 34 类类别映射

| id | 名称 | id | 名称 | id | 名称 |
|---|---|---|---|---|---|
| 0 | Apple | 12 | Citrus | 24 | Potato |
| 1 | Banana | 13 | Coffee | 25 | Raspberry |
| 2 | Basil | 14 | Corn | 26 | Rice |
| 3 | Bean | 15 | Cucumber | 27 | Soybean |
| 4 | Bell pepper | 16 | Eggplant | 28 | Squash |
| 5 | Blueberry | 17 | Garlic | 29 | Strawberry |
| 6 | Broccoli | 18 | Ginger | 30 | Tobacco |
| 7 | Cabbage | 19 | Grape | 31 | Tomato |
| 8 | Carrot | 20 | Lettuce | 32 | Wheat |
| 9 | Cauliflower | 21 | Maple | 33 | Zucchini |
| 10 | Celery | 22 | Peach | | |
| 11 | Cherry | 23 | Plum | | |

- 映射来源：`F:/_plantseg/dataset_plantseg_34/classes.json`（mode=host）与 `data.yaml`，二者逐字一致。

## 3. train / val / test 图片数量

| 划分 | 图片数 | 标签数 |
|---|---|---|
| train | 5367 | 5367 |
| val | 846 | 846 |
| test | 1560 | 1560 |
| **合计** | **7773** | **7773** |

- 空标签图片：0（train/val/test 均为 0）。

## 4. 初始标注集 L0（`al34_random`，即教程中的 `a134_random`）数量与占比

- 图片数：**1443**（labels 同为 1443；`labeled.txt` 去重 1443）。
- 占 train 比例：1443 / 5367 = **26.89%**。
- 与 val/test **零重叠**（无泄漏），是 train 的严格子集。

## 5. 未标注池 U0 数量

- U0 = train − L0 = 5367 − 1443 = **3924**。
- ⚠️ 防泄漏：U0 的标签虽在磁盘存在，但 VARD-AL 选中前**严禁读取**，须经模拟 Oracle 接口按需提供。

## 6. 各类别样本分布（图片数 / 实例数）

| id | 类别 | train图/实例 | val图/实例 | test图/实例 | 总实例 |
|---|---|---|---|---|---|
| 0 | Apple | 274/2176 | 46/414 | 81/509 | 3099 |
| 1 | Banana | 244/1446 | 36/205 | 74/327 | 1978 |
| 2 | Basil | 28/114 | 3/8 | 10/50 | 172 |
| 3 | Bean | 98/427 | 16/69 | 30/112 | 608 |
| 4 | Bell pepper | 116/1026 | 24/275 | 36/406 | 1707 |
| 5 | Blueberry | 97/487 | 18/70 | 27/237 | 794 |
| 6 | Broccoli | 64/1042 | 10/148 | 17/293 | 1483 |
| 7 | Cabbage | 121/1000 | 22/223 | 38/361 | 1584 |
| 8 | Carrot | 74/798 | 11/87 | 20/285 | 1170 |
| 9 | Cauliflower | 39/279 | 6/54 | 8/39 | 372 |
| 10 | Celery | 29/498 | 3/35 | 8/97 | 630 |
| 11 | Cherry | 62/1399 | 11/139 | 18/284 | 1822 |
| 12 | Citrus | 291/2938 | 42/536 | 83/878 | 4352 |
| 13 | Coffee | 135/1418 | 21/225 | 40/444 | 2087 |
| 14 | Corn | 289/1640 | 46/247 | 84/419 | 2306 |
| 15 | Cucumber | 265/1951 | 40/296 | 78/726 | 2973 |
| 16 | Eggplant | 60/679 | 12/115 | 20/151 | 945 |
| 17 | Garlic | 111/809 | 17/119 | 31/207 | 1135 |
| 18 | Ginger | 54/509 | 9/84 | 18/162 | 755 |
| 19 | Grape | 280/1437 | 44/223 | 76/488 | 2148 |
| 20 | Lettuce | 49/176 | 8/15 | 19/56 | 247 |
| 21 | Maple | 40/623 | 7/92 | 11/76 | 791 |
| 22 | Peach | 195/747 | 32/104 | 60/146 | 997 |
| 23 | Plum | 84/257 | 11/47 | 25/82 | 386 |
| 24 | Potato | 103/602 | 16/55 | 27/197 | 854 |
| 25 | Raspberry | 55/383 | 10/73 | 12/20 | 476 |
| 26 | Rice | 78/654 | 6/77 | 21/216 | 947 |
| 27 | Soybean | 356/1766 | 46/374 | 98/497 | 2637 |
| 28 | Squash | 96/170 | 13/22 | 35/63 | 255 |
| 29 | Strawberry | 33/315 | 6/64 | 10/97 | 476 |
| 30 | Tobacco | 97/1062 | 11/231 | 26/364 | 1657 |
| 31 | Tomato | 460/4231 | 74/766 | 133/1336 | 6333 |
| 32 | Wheat | 798/3552 | 137/583 | 231/1080 | 5215 |
| 33 | Zucchini | 192/1885 | 32/379 | 55/344 | 2608 |

- 完整机读版见 `experiments/class_distribution.csv`。
- 不平衡：实例 max/min = 36.8×（Tomato 6333 ↔ Basil 172），CV=0.857；低占比类（<0.5%）：Basil、Lettuce、Squash。

## 7. 数据路径与配置路径

| 项 | 路径 |
|---|---|
| 原始数据根 | `F:/_plantseg/dataset_plantseg_34` |
| 原始 data.yaml | `F:/_plantseg/dataset_plantseg_34/data.yaml` |
| 原始 classes.json | `F:/_plantseg/dataset_plantseg_34/classes.json` |
| AL 初始集 L0 | `F:/_plantseg/_al/al34_random` |
| 本地数据配置 | `configs/dataset_plantseg34.local.yaml` |
| 通用数据配置 | `configs/dataset.yaml` |
| 基线超参配置 | `configs/baseline_yolov8n_seg.yaml` |
| 主动学习配置 | `configs/active_learning.yaml` |

## 8. 数据集许可与来源记录

- **来源**：PlantSeg 公开数据集（大规模真实场景植物病害分割数据集），本项目使用其派生版：
  `dataset_plantseg_34`（宿主级 34 类），由官方 COCO/掩码重建实例，`classes.json` 注明
  “PlantSeg Metadata.csv (Plant/Disease/Index)；实例由 PNG 掩码连通域重建”。
- **标签形态**：YOLO 实例分割多边形（`class_id x1 y1 ...`，归一化）。**本项目未持有原始 PNG 掩码**，
  仅有从掩码派生的多边形标签。
- **许可（License）**：以 PlantSeg 官方发布的许可为准（须在官方仓库/Dataset card 核实后填入准确条款；
  本记录不擅自断言具体许可证，避免误标）。用于学术研究与竞赛，非商业分发。
- **使用约束**：数据不入 Git；不二次分发原始图像；论文需引用 PlantSeg 原始文献。

## 9. 标签来源与 BAGS 监督声明（论文强制说明）

- **本项目无原始 PNG 像素掩码**（已核实 `png_count=0`）。现有多边形标签是由原始 PNG 掩码
  经连通域重建后转换得到的 **YOLO 实例分割多边形**。
- 因此：
  - **YOLOv8n-seg 直接使用现有多边形标签**训练，无需原始掩码。
  - **BAGS 的边界/有符号距离场监督由多边形栅格化生成**，而非读取原始 PNG 像素标签。
  - **论文必须明确说明**：BAGS 监督来自“转换后的多边形标签”，不是“原始 PNG 像素标签”，
    以避免对监督来源的误述。
- **主动学习防泄漏**：U0（未标注池）的标签虽在磁盘存在，但 VARD-AL 选中前严禁读取；
  由模拟 **Oracle 接口**在“选中某图后”才提供对应标签。基线阶段不涉及 U0 标签读取。
