# backend/ —— 标注服务（队友负责）

把 `src/` 提供的能力包装成 HTTP API，供前端调用。接口契约见 [`../docs/INTERFACE.md`](../docs/INTERFACE.md)。

## 待实现

- [ ] `POST /api/predict` —— 调用模型，返回实例列表（class_id / score / polygon / bbox）
- [ ] `POST /api/sample` —— 主动学习选样（random / entropy / tta / coreset）
- [ ] `POST /api/edit-cost` —— 计算预标注→真值的修改次数与拆解
- [ ] `GET  /api/health` —— 健康检查与模型元信息

## 建议技术栈（队友自选）

- Python：FastAPI + Uvicorn（推荐，便于与 `src/` 同语言集成）
- 或 Node.js / Go 等，只要出入参与契约一致即可

## 关键约束

- 单张 `/api/predict` 时延 **≤ 300 ms**（模型侧实测 23–24 ms @ 640）。
- 模型加载在进程启动时完成，避免每次请求重复加载权重（权重不进仓库，见 `../weights/README.md`）。
- 入参图建议先落到本地缓存目录，再交给 `src/` 推理脚本，减少 base64 传输开销。

## 目录约定

```
backend/
├── README.md
├── app.py            # 入口
├── routers/          # /api/predict, /api/sample, /api/edit-cost
├── model_bridge/     # 调用 src/ 的封装（命令行 / 子进程 / 或直接 import）
└── tests/
```
