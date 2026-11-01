# datasets_tools/

数据准备、格式转换与校验脚本。

职责范围（对应旧 `src/` 中的原型脚本，逐步迁移至此）：

| 功能 | 旧脚本 |
|------|--------|
| PlantSeg 数据准备 | `src/plantseg_prepare.py` |
| PlantSeg 标签修正 | `src/plantseg_fix_labels.py` |
| PlantDoc 数据准备 | `src/plantdoc_prepare.py` |
| 数据集巡检 | `src/m0_inspect.py` |
| 大文件迁移到高速盘 | `src/move_to_fast_disk.py` |

约定：输入输出路径统一从 `configs/dataset.yaml` 读取，不在脚本里硬编码。
