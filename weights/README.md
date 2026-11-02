# weights/ —— 模型权重（不进仓库）

模型权重文件（`*.pt` / `*.pth` 等）体积大且有许可限制，**不纳入版本库**（被 `.gitignore` 排除）。
本目录仅作为本地权重存放位置的约定说明。

## 需要放置的权重

| 文件 | 来源 | 用途 |
|------|------|------|
| `yolo11n-seg.pt` | ultralytics 官方 | 在线实例分割基模型 |
| `runs/plantseg34/weights/best.pt` | 本地训练产物 | 当前最优在线模型 |
| `sam2.1_t.pt` | Meta SAM 2.1 | 离线伪标注精修（仅离线用，不参与推理主链路） |

## 本地放置建议

- 训练产物统一输出到仓库外的 `F:/_plantseg/` 或 `runs/`（已 gitignore）。
- 部署时由 backend 在进程启动阶段加载，路径通过环境变量或 `configs/` 配置注入。

## 随仓库分发的基线权重

- `baseline_yolov8nseg_plantseg34_best.pt`
  - 模型：YOLOv8n-seg，34 类（PlantSeg 植物宿主）
  - 训练：150 epoch，imgsz 640，seed 42（`configs/baseline_yolov8n_seg.yaml`）
  - 指标：Box mAP50 0.299 / Mask mAP50 0.287（mAP50-95 Box 0.167 / Mask 0.140）
  - 用途：队友联调预标注接口（`inference/`），非最终精度
  - 复现产物：`runs/plantseg34_baseline/`（未入库）
