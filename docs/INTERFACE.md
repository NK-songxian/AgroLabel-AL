# 接口契约（模型 / 后端 / 前端）

> 本文档定义三方协作的最小接口。模型侧（强哥）产出能力，后端（队友）包装成 HTTP API，
> 前端（队友）调用 API。先冻结契约，再各自实现，避免返工。
>
> ⚠️ **本文只含"模型能力封装"的 4 个接口。标注工作流所需的另外 8 个接口
> （`/api/images`、`/api/refine`、`/api/annotations`、`/api/logs/ops` 等）
> 见 [`INTERFACE_ADDENDUM.md`](INTERFACE_ADDENDUM.md)，该文件已于 2026-11-02 团队确认，
> 与本文**同等有效**。**

## 1. 数据格式约定

- **图像**：PNG / JPG，RGB。
- **分割标注（YOLO seg txt）**：每行 `class_id x1 y1 x2 y2 ...（归一化多边形顶点）`。
- **真值掩码（PlantSeg png）**：像素值 = 类别 Index + 1，背景为 0（PIL 约定）。
- **实例表示**：`{class_id, class_name, score, polygon(归一化), bbox}`，mask 以多边形或 RLE 传递。

## 2. 后端需实现的 API（模型能力封装）

| 方法 | 路径 | 入参 | 出参 |
|------|------|------|------|
| POST | `/api/predict` | `{image_base64 或 image_id}` | `{instances:[{class_id,class_name,score,polygon,bbox}], latency_ms}` |
| POST | `/api/sample` | `{pool:[image_id...], k, strategy(random\|entropy\|tta\|coreset)}` | `{selected:[image_id...], reasons}` |
| POST | `/api/edit-cost` | `{pred_instances, gt_instances}` | `{edit_count, breakdown:{新画,删实例,改类别,拖边界,标为无目标}}` |
| GET | `/api/health` | — | `{status:"ok", model:"yolo11n-seg", classes:34}` |

- 时延预算：**单张 `/api/predict` ≤ 300 ms**（当前实测 23–24 ms @ 640）。
- `/api/sample` 在后端调用模型不确定性/多样性打分，返回待标注图像，供前端优先展示。

## 3. 前端需实现的交互

1. 加载一批图像 → 调 `/api/predict` 显示预标注实例（不同类别不同颜色）。
2. 标注员在预标注基础上做最小化修改（画/删/改类/拖边界/标为无目标）。
3. 修改动作记录为操作日志（供 `m5_logstats.py` / `m6_editcost.py` 计算验收数字）。
4. "智能选样"按钮 → 调 `/api/sample` 拉取模型最不确定 / 最具多样性的一批图，形成主动学习闭环。

## 4. 验收数字由谁产出

- **单次修改次数、漏检/误检、时延**：模型侧脚本（`m6`, `m9`, `m10`）离线验证后，由后端在线上复算。
- **同等标注量下降 ≥20%、同精度修改次数下降 ≥30%**：由 `m8_al.py`（主动学习闭环）产出曲线。

## 5. 联调顺序建议

1. 模型侧先固化 `/api/predict` 的出入参（与本文档表一致）。
2. 后端先用假数据 mock 三个接口，前端并行开发。
3. 后端接真模型，三方联调；以 `out/report/*.json` 中的离线数字为回归基线。
