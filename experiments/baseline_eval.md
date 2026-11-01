# YOLOv8n-seg 基线评估摘要

- 权重: `F:\AgroLabel-AL\runs\baseline5\weights\best.pt`
- 数据: `configs/dataset_plantseg34.local.yaml`
- 可视化: `F:\AgroLabel-AL\experiments\baseline_vis\pred`

| split | Box P  | Box R  | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
| ----- | ------ | ------ | --------- | ------------ | ------ | ------ | ---------- | ------------- |
| val   | 0.2708 | 0.1137 | 0.0603    | 0.0331       | 0.2630 | 0.1081 | 0.0568     | 0.0279        |
| test  | 0.2128 | 0.1118 | 0.0580    | 0.0320       | 0.2098 | 0.1088 | 0.0556     | 0.0274        |

> 注：若为短 epoch 启动基线(v0)，指标偏低属预期；正式 150 epoch 训练后应显著提升。

