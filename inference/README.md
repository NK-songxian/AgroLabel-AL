# inference/

本地推理与预标注服务。

| 功能 | 旧脚本 |
|------|--------|
| 单图/批量推理 | `src/m1_infer.py` |
| 标注应用服务（Gradio/API） | `src/m4_app.py` |
| 预测可视化 | `src/m10_vis_pred.py` |

约定：推理优先走 ONNX Runtime + INT8 量化，单张预标注时延目标 ≤ 300 ms。
