# AgroLabel-AL

> 面向智慧农业场景的多模态数据智能预标注与主动学习原型系统
> 「弘毅杯」第三届桂电南宁研究院「AI+」创新实践大赛 · 企业命题赛道 · 赛题一

本仓库实现一套**人机协同的农业病害实例分割预标注系统**：用离线伪标注 + 在线主动学习，
让标注员以最少的修改次数得到可用的像素级标注。

## 目录结构

```
AgroLabel-AL/
├── datasets/                  # 数据集（原始大文件不进仓库，仅保留结构）
│   ├── PlantVillage/          # 分类数据：images/ + classification_labels.csv
│   ├── PlantSeg/              # 实例分割：images/ masks/ detection_labels/ splits/
│   └── PlantDoc/              # 自然场景：images/ + detection_labels/
├── configs/                   # 配置中心（YAML）
│   ├── dataset.yaml           # 数据集路径与划分
│   ├── model.yaml             # 模型结构与训练超参
│   ├── active_learning.yaml   # 主动学习策略与权重
│   └── default.yaml           # 全项目默认配置（含伪标注参数）
├── models/                    # 模型定义、权重说明（*.pt 不进仓库）
├── datasets_tools/            # 数据准备/转换/校验脚本
├── training/                  # 训练与蒸馏入口
├── active_learning/           # VARD-AL 采样策略与评估
├── inference/                 # 本地推理与预标注服务
├── backend/                   # 后端标注服务（队友负责，含接口契约）
├── frontend/                  # 前端标注界面（队友负责）
├── evaluation/                # 评测脚本（检测/分割/主动学习/提效）
├── experiments/               # 实验配置、日志与结果汇总
├── src/                       # 早期原型脚本（M0–M11，待按新结构迁移）
├── data/                      # 数据说明（历史目录）
├── weights/                   # 权重说明（历史目录）
└── docs/                      # 工程文档与赛题材料
```

## 团队协作分工

| 模块 | 负责人 | 说明 |
|------|--------|------|
| `datasets_tools/` `training/` `active_learning/` `inference/` `evaluation/` | 强哥（Ryan） | 数据准备、训练、评估、主动学习闭环 |
| `backend/` 标注服务 | 队友 | 提供模型推理 / 选样 / 编辑代价 API（契约见 `docs/INTERFACE.md`） |
| `frontend/` 标注界面 | 队友 | 标注员操作界面，调用 backend API |
| `docs/` | 全员 | 架构论证、模块清单、接口契约 |

> `src/` 为早期快速原型脚本（`m0_inspect` ~ `m11_zeroshot_probe`），功能正逐步迁移至
> `datasets_tools/ training/ active_learning/ inference/ evaluation/ experiments/` 五个模块目录。

## 模型侧快速开始（强哥）

```bash
# 建议使用 conda 虚拟环境 sam_gpu（Python 3.11，已含 torch / ultralytics / opencv 等）
conda activate sam_gpu

pip install -r requirements.txt

# 数据准备（PlantSeg → YOLO 分割格式）
python src/plantseg_prepare.py
python src/plantseg_fix_labels.py

# 训练实例分割模型（--data 指向数据集自带的 data.yaml，--config 指向全局配置）
python src/m3_train.py --data F:/_plantseg/dataset_plantseg_34/data.yaml --config configs/default.yaml --name plantseg34_yolo11n

# 主动学习闭环（真实真值当 Oracle）
python src/m8_al.py --strategies random entropy tta coreset
```

> 权重文件（`*.pt`）、`out/`、`runs/` 以及 `datasets/**/images`、`datasets/**/masks`
> 下的原始数据由 `.gitignore` 排除，不在版本库内。

## 分支与协作约定

- `main`：稳定可演示版本，受保护，仅通过 PR 合并。
- `model/*`：模型侧开发分支（强哥）。
- `backend/*`、`frontend/*`：队友开发分支。
- 提交信息使用中文或英文均可，建议带前缀：`[model]`, `[backend]`, `[frontend]`, `[docs]`。

## 关键指标（验收口径）

| 指标 | 目标 |
|------|------|
| 单张预标注时延 | ≤ 300 ms |
| 同等标注质量下所需标注量 | 较随机采样下降 ≥ 20% |
| 同精度下单张修改次数 | 下降 ≥ 30% |
| 预标注精度（mask mAP@0.5） | 越高越好（当前 0.2492） |

## 许可

MIT License，见 `LICENSE`。
