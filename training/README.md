# training/

训练与蒸馏入口。

| 功能 | 旧脚本 |
|------|--------|
| 数据集构建（YOLO 格式） | `src/m3_dataset.py` |
| 实例分割训练 | `src/m3_train.py` |
| 伪标注生成 | `src/m2_pseudo.py` |

约定：
- 超参从 `configs/model.yaml` 读取；
- 每次训练在 `experiments/` 下建立独立目录，记录配置快照与指标曲线；
- 权重输出到 `models/<exp_name>/`，`*.pt` 不入库。
