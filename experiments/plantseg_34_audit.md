# Task-002A · PlantSeg 34 类数据正式审计
> 约束：仅读取，未修改任何数据 / 标签 / 划分，未下载外部数据。所有统计来自真实文件。

## 0. 实际路径记录
- **data_root**: `F:\_plantseg`
- **dataset_dir**: `F:\_plantseg\dataset_plantseg_34`
- **data_yaml**: `F:\_plantseg\dataset_plantseg_34\data.yaml`
- **classes_json**: `F:\_plantseg\dataset_plantseg_34\classes.json`
- **al_dir**: `F:\_plantseg\_al\al34_random`
- **al_labeled_txt**: `F:\_plantseg\_al\al34_random\labeled.txt`
- 数据根目录: `F:\_plantseg`

## 1. 配置文件检查
- `classes.json`: 存在=True, 类别数=34, mode=host, source=PlantSeg Metadata.csv (Plant/Disease/Index)
- `data.yaml`: 存在=True, nc=34, names 数=34, names 与 classes.json 一致=True
  - path 字段: `F:/_plantseg/dataset_plantseg_34`
  - note: 官方 COCO JSON 三划分 image_id 不唯一，实例由 PNG 掩码连通域重建

## 2&5. 划分图片总数
| 划分 | 图片数 | 标签数 |
|---|---|---|| train | 5367 | 5367 |
| val | 846 | 846 |
| test | 1560 | 1560 |
| **合计** | **7773** | |
- 空标签图片数(有图无实例): {'train': 0, 'val': 0, 'test': 0}

## 3. 34 类全覆盖确认
- **结论**: ✅ 34 类在 train/val/test 均存在有效样本
- ✅ train: 34/34 类均有样本
- ✅ val: 34/34 类均有样本
- ✅ test: 34/34 类均有样本

## 4. 逐类别分布（图片数 / 实例数）
| id | 类别 | train图 | train实例 | val图 | val实例 | test图 | test实例 | 总图 | 总实例 |
|---|---|---|---|---|---|---|---|---|---|| 0 | Apple | 274 | 2176 | 46 | 414 | 81 | 509 | 401 | 3099 |
| 1 | Banana | 244 | 1446 | 36 | 205 | 74 | 327 | 354 | 1978 |
| 2 | Basil | 28 | 114 | 3 | 8 | 10 | 50 | 41 | 172 |
| 3 | Bean | 98 | 427 | 16 | 69 | 30 | 112 | 144 | 608 |
| 4 | Bell pepper | 116 | 1026 | 24 | 275 | 36 | 406 | 176 | 1707 |
| 5 | Blueberry | 97 | 487 | 18 | 70 | 27 | 237 | 142 | 794 |
| 6 | Broccoli | 64 | 1042 | 10 | 148 | 17 | 293 | 91 | 1483 |
| 7 | Cabbage | 121 | 1000 | 22 | 223 | 38 | 361 | 181 | 1584 |
| 8 | Carrot | 74 | 798 | 11 | 87 | 20 | 285 | 105 | 1170 |
| 9 | Cauliflower | 39 | 279 | 6 | 54 | 8 | 39 | 53 | 372 |
| 10 | Celery | 29 | 498 | 3 | 35 | 8 | 97 | 40 | 630 |
| 11 | Cherry | 62 | 1399 | 11 | 139 | 18 | 284 | 91 | 1822 |
| 12 | Citrus | 291 | 2938 | 42 | 536 | 83 | 878 | 416 | 4352 |
| 13 | Coffee | 135 | 1418 | 21 | 225 | 40 | 444 | 196 | 2087 |
| 14 | Corn | 289 | 1640 | 46 | 247 | 84 | 419 | 419 | 2306 |
| 15 | Cucumber | 265 | 1951 | 40 | 296 | 78 | 726 | 383 | 2973 |
| 16 | Eggplant | 60 | 679 | 12 | 115 | 20 | 151 | 92 | 945 |
| 17 | Garlic | 111 | 809 | 17 | 119 | 31 | 207 | 159 | 1135 |
| 18 | Ginger | 54 | 509 | 9 | 84 | 18 | 162 | 81 | 755 |
| 19 | Grape | 280 | 1437 | 44 | 223 | 76 | 488 | 400 | 2148 |
| 20 | Lettuce | 49 | 176 | 8 | 15 | 19 | 56 | 76 | 247 |
| 21 | Maple | 40 | 623 | 7 | 92 | 11 | 76 | 58 | 791 |
| 22 | Peach | 195 | 747 | 32 | 104 | 60 | 146 | 287 | 997 |
| 23 | Plum | 84 | 257 | 11 | 47 | 25 | 82 | 120 | 386 |
| 24 | Potato | 103 | 602 | 16 | 55 | 27 | 197 | 146 | 854 |
| 25 | Raspberry | 55 | 383 | 10 | 73 | 12 | 20 | 77 | 476 |
| 26 | Rice | 78 | 654 | 6 | 77 | 21 | 216 | 105 | 947 |
| 27 | Soybean | 356 | 1766 | 46 | 374 | 98 | 497 | 500 | 2637 |
| 28 | Squash | 96 | 170 | 13 | 22 | 35 | 63 | 144 | 255 |
| 29 | Strawberry | 33 | 315 | 6 | 64 | 10 | 97 | 49 | 476 |
| 30 | Tobacco | 97 | 1062 | 11 | 231 | 26 | 364 | 134 | 1657 |
| 31 | Tomato | 460 | 4231 | 74 | 766 | 133 | 1336 | 667 | 6333 |
| 32 | Wheat | 798 | 3552 | 137 | 583 | 231 | 1080 | 1166 | 5215 |
| 33 | Zucchini | 192 | 1885 | 32 | 379 | 55 | 344 | 279 | 2608 |

## 6. AL 初始标注集占比 (al34_random)
- AL 子集图片数: 1443，标签数: 1443
- `dataset_plantseg_34/train` 总数: 5367
- **占比**: 26.89%
- 是否为 train 子集: True（不在 train 中: 0 个）
- `labeled.txt` 去重行数: 1443；图不在列表: 0；列表无图: 0

## 7. AL 与 val/test 重叠
- 与 val 重叠: 0 个
- 与 test 重叠: 0 个
- **结论**: ✅ 完全不重叠

## 8. 未标注池 U0
- U0 = |train| - |al34_random| = 5367 - 1443 = 3924
- **U0 规模 = 3924**

## 9. 类别不平衡诊断
**按总实例数**
- min=172, max=6333, median=1152.5, mean=1647.0, std=1411.7, CV=0.857, max/min 比=36.82
**按 train 实例数**
- min=114, max=4231, median=803.5, mean=1132.2, std=950.8, CV=0.84, max/min 比=37.11
- **极度不平衡标记 (max/min>10)**: True
- 稀有类(总实例<30): 无
- 极低占比类(占比<0.5%): ['Basil', 'Lettuce', 'Squash']

## 修改文件列表
- 仅生成以下审计产物（未修改任何原始数据 / 标签 / 配置）：
  - `F:\AgroLabel-AL\experiments\plantseg_34_audit.md`
  - `F:\AgroLabel-AL\experiments\plantseg_34_audit.json`
  - `F:\AgroLabel-AL\experiments\class_distribution.csv`
