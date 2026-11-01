# models/

模型定义与权重说明目录。

- 模型结构定义（AgroHAFM-Seg、HAFM、BAGS、语义原型分支）放在本目录。
- 训练产物权重（`*.pt` / `*.onnx` / `*.engine`）**不进仓库**，由 `.gitignore` 排除。
- 建议按 `models/<exp_name>/best.pt` 组织，并在 `experiments/` 记录对应配置与指标。
