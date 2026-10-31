# src/ 模块清单

> 模型与算法全部位于 `src/` 目录。脚本按"阶段 + 模块号"编号（M0–M11），方便对照实施教程。

| 文件 | 阶段 | 职责 |
|------|------|------|
| `common.py` | — | 公共函数：配置加载、图片遍历、类别解析、可视化 |
| `m0_inspect.py` | M0 | 数据体检：把原始数据集变成一张可查的表 |
| `plantseg_prepare.py` | 数据 | 将 PlantSeg（像素级病斑掩码）整理为 YOLO 分割格式 |
| `plantseg_fix_labels.py` | 数据 | 定点修复"掩码值 ≠ Index+1"的十几张图的标签 |
| `plantdoc_prepare.py` | 数据 | 将 PlantDoc（仅有框、无掩码）整理为 YOLO 检测格式 |
| `move_to_fast_disk.py` | 数据 | 将派生数据集从慢盘 H: 搬到快盘 F:，消除训练 IO 瓶颈 |
| `m1_infer.py` | M1/M3 | 通用推理脚本：跑模型、出可视化、顺手量时延 |
| `m2_pseudo.py` | M2 | 离线伪标注：从分类标签造出叶片实例掩码与 YOLO 分割标签 |
| `m3_dataset.py` | M3 | 把伪标签整理成 ultralytics 可直接读取的目录结构 |
| `m3_train.py` | M3 | 用伪标签训练第一个实例分割模型（扩展版，支持 eval-only / 多种增强） |
| `m4_app.py` | M4 | 人机协同标注界面（Gradio demo） |
| `m5_logstats.py` | M5 | 把人机协同操作日志算成验收要的数字 |
| `m6_editcost.py` | M5 | 将"模型预测 vs 真值"折算为"人工要改几下"（新画/删/改类/拖边界/标为无目标） |
| `m7_longtail.py` | M7 | 把逐类指标拆成"头部 vs 尾部"，产出小样本表现证据表 |
| `m8_al.py` | M8 | 主动学习闭环：用真值当 Oracle，实测同等精度下能省多少标注量 |
| `m9_plantdoc_eval.py` | M9 | PlantDoc 补充评估：跨域鲁棒性 + 真实分辨率下时延 |
| `m10_vis_pred.py` | 评测 | 模型预测 vs 人工真值四宫格对比图（命中/漏检/误检诊断） |
| `m11_zeroshot_probe.py` | 论证 | 零样本（开放词表）预标注能力探针，含管线自检（见 ARCHITECTURE.md） |
| `m11b_zeroshot_fig.py` | 论证 | 将探针结果画成可进报告的对比图 |

## 关键数据产物（位于 `out/`，不进仓库）

- `out/zeroshot_probe.json`：零样本对照实验原始数据。
- `out/report/*.json`：mAP、编辑代价、主动学习曲线、PlantDoc 评估等结构化结果。
- `out/figures/*.png`：各类可视化（含 `pred_vs_gt.png`、`zeroshot_vs_trained.png`）。

## 约定

- 所有脚本以 `E:/anaco/envs/sam_gpu/python.exe` 运行（环境路径见 `requirements.txt`）。
- 真值数据集（PlantSeg / PlantDoc）须本地合规存放；脚本只读取，不随仓库分发。
