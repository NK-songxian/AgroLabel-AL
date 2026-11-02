# SAM3 数据语义核查报告（Task-SAM3-001 · 一）

- 数据集：`F:/_plantseg/dataset_plantseg_34`
- 元数据：`F:/_plantseg/dataset_plantseg_115`
- 划分：train 5367 / val 846 / test 1560 张

## 1. 语义结论

| 项 | 结论 |
|---|---|
| 类别层级 | **寄主植物（host）**，34 类，**不是病害类别** |
| 实例标注目标 | **病斑区域（lesion）**，不是整叶/整果 |
| 单实例面积占比 | 中位 **0.0025**，p10 0.0003，p90 0.0561 |
| 标注行格式异常 | 0 行 |
| 多类别混图 | 0 张 |

判定依据：

- 单实例掩码面积占全图中位数 0.0025，p90 0.0561 —— 远小于整叶/整果（通常 >0.15）
- 样例叠加图显示标注贴合果实/叶片上的坏死斑块，未包住整片叶子
- classes.json 标注 mode=host，115 类版本为 Plant___Disease 组合，可直接取 Disease 字段

> **纠正**：此前依据类名认为“34 类无病斑、病斑提示词不适用”，是只看类名未看标注内容；标注目标确为病斑，病斑级提示词成立。

## 2. 划分统计

| 划分 | 图像 | 实例 | 无实例图 |
|---|---|---|---|
| train | 5367 | 38496 | 0 |
| val | 846 | 6454 | 0 |
| test | 1560 | 11049 | 0 |

## 3. 逐类分布

| id | 类别 | train | val | test | 实例合计 | 面积中位 | 主要病害 |
|---|---|---|---|---|---|---|---|
| 0 | Apple | 274 | 46 | 81 | 3099 | 0.0017 | apple scab、apple rust |
| 1 | Banana | 244 | 36 | 74 | 1978 | 0.0036 | banana anthracnose、banana black leaf streak |
| 2 | Basil | 28 | 3 | 10 | 172 | 0.0075 | basil downy mildew |
| 3 | Bean | 98 | 16 | 30 | 608 | 0.0120 | bean halo blight、bean rust |
| 4 | Bell pepper | 116 | 24 | 36 | 1707 | 0.0011 | bell pepper bacterial spot、bell pepper frogeye leaf spot |
| 5 | Blueberry | 97 | 18 | 27 | 794 | 0.0058 | blueberry rust、blueberry scorch |
| 6 | Broccoli | 64 | 10 | 17 | 1483 | 0.0010 | broccoli alternaria leaf spot、broccoli downy mildew |
| 7 | Cabbage | 121 | 22 | 38 | 1584 | 0.0031 | cabbage black rot、cabbage downy mildew |
| 8 | Carrot | 74 | 11 | 20 | 1170 | 0.0027 | carrot alternaria leaf blight、carrot cavity spot |
| 9 | Cauliflower | 39 | 6 | 8 | 372 | 0.0051 | cauliflower alternaria leaf spot、cauliflower bacterial soft rot |
| 10 | Celery | 29 | 3 | 8 | 630 | 0.0015 | celery early blight、celery anthracnose |
| 11 | Cherry | 62 | 11 | 18 | 1822 | 0.0003 | cherry leaf spot、cherry powdery mildew |
| 12 | Citrus | 291 | 42 | 83 | 4352 | 0.0019 | citrus canker、citrus greening disease |
| 13 | Coffee | 135 | 21 | 40 | 2087 | 0.0010 | coffee leaf rust、coffee berry blotch |
| 14 | Corn | 289 | 46 | 84 | 2306 | 0.0041 | corn gray leaf spot、corn northern leaf blight |
| 15 | Cucumber | 265 | 40 | 78 | 2973 | 0.0024 | cucumber angular leaf spot、cucumber powdery mildew |
| 16 | Eggplant | 60 | 12 | 20 | 945 | 0.0016 | eggplant cercospora leaf spot、eggplant phomopsis fruit rot |
| 17 | Garlic | 111 | 17 | 31 | 1135 | 0.0023 | garlic rust、garlic leaf blight |
| 18 | Ginger | 54 | 9 | 18 | 755 | 0.0011 | ginger leaf spot、ginger sheath blight |
| 19 | Grape | 280 | 44 | 76 | 2148 | 0.0061 | grape black rot、grape downy mildew |
| 20 | Lettuce | 49 | 8 | 19 | 247 | 0.0204 | lettuce downy mildew、lettuce mosaic virus |
| 21 | Maple | 40 | 7 | 11 | 791 | 0.0016 | maple tar spot |
| 22 | Peach | 195 | 32 | 60 | 997 | 0.0109 | peach leaf curl、peach scab |
| 23 | Plum | 84 | 11 | 25 | 386 | 0.0207 | plum pox virus、plum pocket disease |
| 24 | Potato | 103 | 16 | 27 | 854 | 0.0047 | potato early blight、potato late blight |
| 25 | Raspberry | 55 | 10 | 12 | 476 | 0.0013 | raspberry leaf spot、raspberry yellow rust |
| 26 | Rice | 78 | 6 | 21 | 947 | 0.0014 | rice blast、rice sheath blight |
| 27 | Soybean | 356 | 46 | 98 | 2637 | 0.0035 | soybean frog eye leaf spot、soybean mosaic |
| 28 | Squash | 96 | 13 | 35 | 255 | 0.2133 | squash powdery mildew |
| 29 | Strawberry | 33 | 6 | 10 | 476 | 0.0028 | strawberry leaf scorch、strawberry anthracnose |
| 30 | Tobacco | 97 | 11 | 26 | 1657 | 0.0008 | tobacco brown spot、tobacco blue mold |
| 31 | Tomato | 460 | 74 | 133 | 6333 | 0.0033 | tomato early blight、tomato bacterial leaf spot |
| 32 | Wheat | 798 | 137 | 231 | 5215 | 0.0068 | wheat stripe rust、wheat head scab |
| 33 | Zucchini | 192 | 32 | 55 | 2608 | 0.0009 | zucchini powdery mildew、zucchini downy mildew |

- 无实例类别：无

## 4. 抽样可视化（每类 ≥3 张）

输出目录：`experiments/sam3_sample_vis/`（左=原图，中=GT 掩码，右=多边形+框+类别）

- `experiments/sam3_sample_vis/00_Apple_0.jpg` — Apple / apple scab / 1 实例（train）
- `experiments/sam3_sample_vis/00_Apple_1.jpg` — Apple / apple black rot / 7 实例（train）
- `experiments/sam3_sample_vis/00_Apple_2.jpg` — Apple / apple scab / 2 实例（train）
- `experiments/sam3_sample_vis/01_Banana_0.jpg` — Banana / banana black leaf streak / 1 实例（train）
- `experiments/sam3_sample_vis/01_Banana_1.jpg` — Banana / banana black leaf streak / 2 实例（train）
- `experiments/sam3_sample_vis/01_Banana_2.jpg` — Banana / banana panama disease / 1 实例（train）
- `experiments/sam3_sample_vis/02_Basil_0.jpg` — Basil / basil downy mildew / 8 实例（train）
- `experiments/sam3_sample_vis/02_Basil_1.jpg` — Basil / basil downy mildew / 5 实例（train）
- `experiments/sam3_sample_vis/02_Basil_2.jpg` — Basil / basil downy mildew / 4 实例（train）
- `experiments/sam3_sample_vis/03_Bean_0.jpg` — Bean / bean rust / 2 实例（train）
- `experiments/sam3_sample_vis/03_Bean_1.jpg` — Bean / bean halo blight / 2 实例（train）
- `experiments/sam3_sample_vis/03_Bean_2.jpg` — Bean / bean rust / 1 实例（train）

共 102 张，完整清单见 `sam3_data_audit.json` → `visual_samples`。
