# AgroLabel-AL 队友交接包（2026-11-17）

> 交接人：强哥 · 当前进度见 `docs/模型实验作战手册.html`（20+ 条实测台账）
> 本包目的：让队友在本地跑通"预标注 + 人工编辑 + 主动学习"全链路，并开展各自分工。

## 一、包内容

| 文件 | 说明 |
|---|---|
| `weights/b6_y26x_1280_best.pt` | **当前最优模型**（YOLO26x-seg @1280，val mask mAP50 0.416 / 50-95 0.214，B5 基线 0.397）。142MB，GPU 推理约 144ms/图 |
| `docs/INTERFACE.md`、`docs/INTERFACE_ADDENDUM.md` | 前后端接口契约（前端已在用，改动需同步） |
| `docs/模型实验作战手册.html` | 全部实验台账（哪些路线已测死、哪些有效） |
| `docs/al_effect_smoke.png` | 主动学习效果可视化（汇报可直接引用） |

## 二、本地跑通三步（Windows）

```bash
# 1. 环境（Python 3.10+，需 NVIDIA GPU）
pip install fastapi uvicorn ultralytics opencv-python pyyaml pillow numpy

# 2. 数据集（约 7773 图，二选一）
#    a) 向强哥要 plantseg34 数据集打包（images/train|val|test + labels/）
#    b) 若已克隆仓库，数据在 F:/_plantseg/dataset_plantseg_34（本地机）

# 3. 启动后端（在仓库根目录；数据/权重路径用环境变量切换，不改代码）
set AGROLABEL_DATA_ROOT=<数据集根目录>
set AGROLABEL_WEIGHTS=<解压后的 b6_y26x_1280_best.pt 绝对路径>
set AGROLABEL_IMGSZ=1280
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000

# 前端
cd frontend && npm install && npm run dev   # 打开 http://localhost:5173/（注意必须用 localhost 或加 --host）
```

## 三、验证后端就绪

- `http://127.0.0.1:8000/api/health` → 应返回 `"model":"b6_y26x_1280_best.pt"`、`"imgsz":1280`
- 页面打开任意图 → 红色多边形掩码 + 右侧实例置信度，即为 B6 预标注生效

## 四、接口要点（踩过的坑，勿重复踩）

1. `image_id` **不带扩展名**（catalog id = 文件 stem），带 `.jpg` 会 404；
2. `/api/sample` 单次 pool 上限 500 张，全池 7773 张需分批；
3. 前端 Vite 默认只听 IPv6，启动加 `--host` 才能用 127.0.0.1 访问；
4. 预测延迟 ~120–260ms/图（GPU），CI/低配机记得降级 `AGROLABEL_DEVICE=cpu`（会慢 10 倍以上）。

## 五、分工建议与验收标准

| 方向 | 任务 | 验收 |
|---|---|---|
| 前端 | 掩码编辑体验（顶点拖拽/删除/改类）、编辑成本上报准确性 | 编辑操作四分类（新增/删除/拖边/改类）计数与后端日志一致 |
| 标注/评测 | 用 B6 预标注做 50 图人工修正计时，对照纯手工 | 单张加权编辑成本 < 纯手工 7.63（B0 基线） |
| AL 演示 | "最不确定"排序截图 + 置信度滑条演示录屏 | 能说清 entropy 采样为何选这几张（score 在 reasons 里） |

## 六、当前主线状态（避免重复劳动）

- **已测死**：SAM3 全参微调（太慢）、零样本全家桶（World/YOLOE/GDINO+SAM2 全部 mAP≈0）、语义投票融合（两代票源均阴性）
- **在跑**：Mask2Former-v2（云端，修正 80 类 bug 后重训）
- **有效主线**：B6@1280 全监督 + AL 闭环（本包模型即主线产物）
