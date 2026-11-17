# Grounded-SAM-2 与 CVAT 研究笔记（2026-11-17）

> 背景：外部调研建议引入 Grounded-SAM-2 作开放词汇定位对照、参考 CVAT 标注平台。
> 本文以项目已有实测证据为基准逐项核实，给出接入决策。

## 一、Grounded-SAM-2（IDEA-Research/Grounded-SAM-2）

### 1.1 仓库核实结论

| 项 | 核实结果 |
|---|---|
| 定位+掩码组合 | Grounding DINO（1.0 本地 / 1.5+付费API / Florence-2）→ SAM2 |
| 免编译路线 | **HF transformers 加载 `IDEA-Research/grounding-dino-tiny` 无需编译 Deformable Attention CUDA 扩展**（本地 GDINO 1.0 需编译，CUDA 12.1 亲和，我们云端是 12.8 有风险） |
| 权重体积 | grounding-dino-tiny ≈0.7G + sam2-hiera-small ≈0.2G（large ≈0.9G），HF 镜像可下 |
| 推理显存 | 冒烟口径 ≈6G（tiny+small），可与训练并行 |
| 文本提示 | 1.0 支持多短语列表（". " 分隔），输出带短语文本回映射类别；Florence-2 多类用 `<and>` |
| 自动标注 demo | `grounded_sam2_florence2_autolabel_pipeline.py`（caption→phrase grounding 级联） |
| JSON 输出 | **非标准 COCO JSON**（自定义结构，RLE 掩码），接 COCOeval 需自行转换 |
| SAHI 切片 | **仅 GDINO 1.5（付费 API）支持**，本地 1.0 路线不含；密集小目标切片路线被锁在 API 后面 |
| 许可证 | SAM2 Apache-2.0、Grounding DINO Apache-2.0；组合仓库 LICENSE 需在采用前复核 |
| 官方环境 | torch≥2.3.1 / CUDA 12.1（我们 torch 2.8+cu128，HF 路线无碍） |

### 1.2 与已有实测证据的对照（期望管理）

| 已测路线 | 结果 | 含义 |
|---|---|---|
| Z2/Z2b YOLO-World、Z3 YOLOE 零样本 | box 0.0038 / 0.0069，7 族提示恒等 | 开放词汇检测在该域近似失效 |
| Z1 SAM3 纯文本提示 | mask 0.0007–0.0102 | 提示敏感、定位近零 |
| F4 SAM3 GT 框提示 | mask 0.8439（oracle） | **瓶颈在定位而非分割** |
| F4-yolo SAM3 YOLO 框提示 | 0.0343 | 定位质量决定最终精度 |

Grounding DINO 在 COCO/开放词汇榜单强于 YOLO-World 一代，但 PlantSeg 病斑/宿主属长尾专业域，**先验概率不应高**。本实验的价值在于：若 GDINO 定位显著优于 YOLO-World（如 box AP50 提升 5 倍以上），"农业专用定位器 + SAM 系分割"的 F4 链路天花板才值得重估；否则进一步坐实"开放词汇路线不可用、全监督 + AL 是唯一主线"的结论。

### 1.3 冒烟实验结果（2026-11-17 已执行，tools_local/gsam2_smoke.py）

| 模式 | 框数 | 存活掩码（area≥64px²） | mask mAP50 | mAP50-95 |
|---|---|---|---|---|
| host（34 宿主短语列表） | 280 | 8 | **0.0000** | 0.0000 |
| lesion（病斑单类短语） | 296 | 13 | **0.0000** | 0.0000 |

口径：val 前 50 图、conf 0.25、grounding-dino-tiny + sam2-hiera-small、COCOeval segm（与 Z 系列同口径）。

**解读**：
1. 两种提示模式均完全失效（0.0000），低于 Z2/Z3 的 0.004–0.007——开放词汇路线第三点证据，且是最强的一点（GDINO 是该系最强开源定位器）。
2. 280/296 个框中仅 8/13 个掩码面积 ≥64px²（8×8 像素）：GDINO 输出大量"斑点级"微小框，与 GT 实例（完整病灶区域）尺度完全不匹配。
3. 结论：**"农业专用定位器（全监督）+ SAM 系分割"是唯一有竞争力的 F4 型链路**；开放词汇定位在 PlantSeg 域不可用，写入手册 Z4 行，Z 系列证据链闭环。

### 1.4 决策

- **做一次冒烟**（成本 ≈0.5h），结果写入手册 Z4 行；
- 不投入更多：SAHI 密集小目标路线锁在付费 API、自动标注 demo 依赖 Florence-2（农业域无微调证据）；
- 比赛报告引用价值：作为"开放词汇路线第三点证据"补全 Z 系列。

## 二、CVAT（cvat-ai/cvat）

### 2.1 仓库核实结论

| 项 | 核实结果 |
|---|---|
| 许可证 | 社区版 MIT（`/serverless` 预置模型可能含非商业第三方资源，采用前逐个审） |
| 功能边界 | 社区版含：掩码/多边形标注、任务/作业管理、多人协作、审核、Honeypot 质检、Grafana 统计、Nuclio serverless 自动标注、REST API + Python SDK（`cvat-sdk`）、20+ 格式导入导出、Webhooks |
| **不含** | 内置 SAM2/SAM3 自动标注、AI agents、SSO——均为付费 Online/Enterprise |
| SAM 交互标注 | 社区版可经 Nuclio 部署 **SAM（一代）** 交互标注；SAM2/SAM3 需自己写 serverless 函数 |
| 部署 | docker compose 全家桶（server/ui/workers/postgres/redis/opa），默认占用 8080，持续运行 |
| 资源 | 官方未列最低配；实测普遍建议 ≥4C8G；GPU 仅 serverless 模型推理需要 |

### 2.2 与 AgroLabel-AL 现状的适配分析

1. **不替换自研前端**：Vue3 + FastAPI 已承载接口约定（INTERFACE.md），队友前端活跃迭代中；替换为 CVAT 引入 Docker 全家桶运维与样式割裂，比赛演示反而扣分。
2. **云端（AutoDL）不适合部署 CVAT**：训练机不保证 8080 长期暴露、磁盘/内存与训练竞争、实例随用随关。
3. **本地部署可选但收益有限**：Windows Docker Desktop 可跑社区版，但 SAM 交互需自写 Nuclio 函数接我们的 SAM3 权重（工程量 ≈ 我们已实现的人工修正链路），而 I-03/I-04 已证明当前预标注质量下人工修正收益微小（-2.6%）。

### 2.3 决策：借鉴三件事，不部署

| 借鉴点 | 落地方式 |
|---|---|
| 任务/作业状态机（project→task→job，assignee/进度/审核流） | 对照我们 FastAPI 的标注任务模型，补齐审核（accepted/rejected）字段即可 |
| Honeypot 质检（埋已知答案样本监控标注员质量） | 写入报告作为"质量控制设计"章节（无需实现） |
| REST API 语义（COCO 导入导出、webhook 事件） | 我们 API 已有 COCO 通道；webhook 事件（标注完成触发 AL 采样）值得补进 INTERFACE.md |

## 三、总决策表

| 项 | 动作 | 理由 |
|---|---|---|
| Grounded-SAM-2 | **冒烟已完成（Z4 入册），不再投入** | 双提示模式 mAP50=0.0000，开放词汇证据链闭环 |
| CVAT | 不部署，借鉴任务状态机 + webhook 设计 | 自研前端已够用，部署收益<成本 |
