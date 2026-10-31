# data/ —— 数据说明（原始大文件不进仓库）

本目录只放**数据准备脚本与说明**，不存放原始图像与权重（被 `.gitignore` 排除）。

## 数据集

| 数据集 | 类型 | 用途 | 是否随仓库 |
|--------|------|------|------------|
| PlantVillage | 分类（仅文件夹名，无掩码） | 离线伪标注源 | 否，需本地合规存放 |
| PlantSeg | 像素级病斑掩码（34 主机 / 115 病害组合） | 真值微调与评测（主战场） | 否，需本地合规存放 |
| PlantDoc | 叶片级检测框（无掩码） | 跨域鲁棒性补充评估 | 否，需本地合规存放 |

## 准备流程（强哥，见 `src/`）

```bash
python src/plantseg_prepare.py      # PlantSeg → YOLO 分割格式
python src/plantseg_fix_labels.py   # 修复偏移标签
python src/plantdoc_prepare.py      # PlantDoc → YOLO 检测格式
```

## 合规提示

- 原始数据集受各自许可约束（如 PlantSeg 为 CC BY-NC 4.0），**不得随仓库分发**。
- 仓库内仅保留脚本、配置与结果 JSON；原始图与权重放本地 `F:/_plantseg/` 等目录。
