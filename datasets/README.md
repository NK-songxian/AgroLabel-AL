# datasets/

数据目录。**原始图像/掩码等大文件不入库**，仓库仅保留目录结构与标签文件；
`.gitignore` 已忽略 `datasets/**/images/*` 与 `datasets/**/masks/*`（保留 `.gitkeep`）。

| 子目录 | 数据集 | 用途 |
|--------|--------|------|
| `PlantVillage/` | PlantVillage | 分类预训练（无掩码） |
| `PlantSeg/` | PlantSeg | 实例分割主数据集（伪实例标签） |
| `PlantDoc/` | PlantDoc | 自然场景泛化评估 |

各子目录布局：

```
PlantVillage/
├── images/                    # 叶片图像（不入库）
└── classification_labels.csv  # 图像 → 类别 标签表

PlantSeg/
├── images/                    # 原图
├── masks/                     # 病斑掩码
├── detection_labels/          # YOLO 格式标注（txt）
└── splits/                    # 官方/自建 train/val/test 划分

PlantDoc/
├── images/
└── detection_labels/
```

> 说明：PlantSeg 的连通区域不等于真实病斑实例，作为**伪实例标签**使用，须抽样人工核验。
> 提取与转换脚本见 `datasets_tools/`。
