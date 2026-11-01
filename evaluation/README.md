# evaluation/

评测脚本（检测 / 分割 / 主动学习 / 提效）。

| 功能 | 旧脚本 |
|------|--------|
| PlantDoc 跨域评估 | `src/m9_plantdoc_eval.py` |
| 日志统计 | `src/m5_logstats.py` |
| 编辑代价（提效）评估 | `src/m6_editcost.py` |

评测口径必须与赛题一致：mask mAP@0.5、同等质量下标注量下降 ≥20%、同精度下修改次数下降 ≥30%。
