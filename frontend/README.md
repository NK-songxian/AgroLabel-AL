# frontend/ —— 标注界面（队友负责）

标注员操作界面，调用 [`../backend`](../backend) 的 API。接口契约见 [`../docs/INTERFACE.md`](../docs/INTERFACE.md)。

## 核心交互（来自契约）

1. 加载一批图像 → 显示 `/api/predict` 返回的预标注实例（不同类别不同颜色）。
2. 标注员在预标注基础上做最小化修改：画 / 删 / 改类别 / 拖边界 / 标为无目标。
3. 修改动作记录操作日志（供后端/模型侧计算验收数字）。
4. "智能选样"按钮 → 调 `/api/sample` 拉取模型最不确定 / 最具多样性的一批图，形成主动学习闭环。

## 待实现

- [ ] 图像画布与多边形标注工具（画/拖/删/改类）
- [ ] 预标注结果叠加显示与一键采纳
- [ ] 操作日志埋点（修改类型 + 耗时）
- [ ] 智能选样入口
- [ ] 与 backend 的联调

## 建议技术栈（队友自选）

- React / Vue + Canvas 或 SVG 标注库（如 Konva、Fabric.js、Label Studio 前端）。
- 保持与契约一致的字段命名（class_id / score / polygon / bbox）。

## 目录约定

```
frontend/
├── README.md
├── package.json
├── src/
│   ├── api/        # 调 backend 的封装
│   ├── components/ # 标注画布、工具栏
│   └── pages/
└── public/
```
