# smart-agri-annot

> 面向智慧农业场景的多模态数据智能预标注与主动学习原型系统
> 「弘毅杯」第三届桂电南宁研究院「AI+」创新实践大赛 · 企业命题赛道 · 赛题一

本仓库实现一套**人机协同的农业病害实例分割预标注系统**：用离线伪标注 + 在线主动学习，
让标注员以最少的修改次数得到可用的像素级标注。

## 团队协作分工

| 模块 | 负责人 | 说明 |
|------|--------|------|
| `src/` 模型与算法 | 强哥（Ryan） | 数据准备、训练、评估、主动学习闭环 |
| `backend/` 标注服务 | 队友 | 提供模型推理 / 选样 / 编辑代价 API（契约见 `docs/INTERFACE.md`） |
| `frontend/` 标注界面 | 队友 | 标注员操作界面，调用 backend API |
| `docs/` | 全员 | 架构论证、模块清单、接口契约 |

## 目录结构

```
smart-agri-annot/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── configs/              # 数据集与训练配置（YAML）
├── src/                  # 模型与算法（强哥负责）
├── backend/             # 后端标注服务（队友负责，含接口契约）
├── frontend/            # 前端标注界面（队友负责）
├── data/                # 数据说明与脚本（原始大文件不进仓库）
├── weights/             # 模型权重（不进仓库，仅留说明）
└── docs/                # 工程文档
    ├── ARCHITECTURE.md  # 架构选型与"为什么不用零样本"的定量论证
    ├── MODULES.md       # src/ 模块清单与角色
    └── INTERFACE.md     # 模型 / 后端 / 前端 三方接口契约
```

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

> 权重文件（`*.pt`）、`out/`、`runs/` 由 `.gitignore` 排除，不在版本库内。

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
