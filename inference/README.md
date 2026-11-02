# inference/

本地推理与预标注服务。

| 功能 | 旧脚本 |
|------|--------|
| 单图/批量推理 | `src/m1_infer.py` |
| 标注应用服务（Gradio/API） | `src/m4_app.py` |
| 预测可视化 | `src/m10_vis_pred.py` |

约定：推理优先走 ONNX Runtime + INT8 量化，单张预标注时延目标 ≤ 300 ms。

## 入口（已实现）

`inference/predict.py` 提供交接文档 §6 约定的签名，后端直接 import：

```python
from inference.predict import predict, predict_response, refine, model_version

instances = predict(image_path)          # list[{class_id,class_name,score,polygon,bbox}]，坐标归一化
resp = predict_response(image_path)      # 额外带 image_width/height、latency_ms、model_version
ref  = refine(image_path, point=[x, y])  # ROI 重识别；point 为原图像素，输出 polygon ≤64 顶点
```

- `predict()` / `refine()` 用线程锁串行化推理；权重默认 `weights/baseline_*.pt`，
  也可设 `AGROLABEL_WEIGHTS` 或传 `weights=` 覆盖。
- `refine()` 是 `INTERFACE_ADDENDUM.md` §2.6 的**参考实现**，后端可直接调，也可自行裁图后调 `predict()`。
  注意：默认半径 `0.08×min(W,H)` 对大病灶可能偏小（实测会 `found=false`），
  前端建议按视口传 `radius`。

