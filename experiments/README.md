# experiments/

实验配置、日志与结果汇总。

| 功能 | 旧脚本 |
|------|--------|
| 零样本探针实验 | `src/m11_zeroshot_probe.py` |
| 零样本结果作图 | `src/m11b_zeroshot_fig.py` |

约定：每个实验一个子目录，包含
- `config_snapshot.yaml`（本次训练/评估的完整配置快照）
- `metrics.md`（关键指标）
- `figures/`（曲线与可视化）

用于支撑技术报告中的消融、对比、主动学习与提效四类证据。
