# AgroLabel-AL 队友交接包（v1.1，2026-11-18）

> 交接人：强哥 · 当前进度见 `docs/模型实验作战手册.html`
> 本包目的：让队友在本地跑通“预标注 + 人工编辑 + 主动学习”全链路，并开展分工实验。
> **v1.1 变更**：三臂实验改为 `manual / assisted / noprompt`；AL 排序移出三臂表；Mask2Former-v2 已收官且阴性。

## 一、三个交付包分别做什么

| 包 | 用途 | 是否含模型 |
|---|---|---|
| `AgroLabel_B6_handoff.zip` | 前后端预标注演示、三臂实验中的 B6 底稿 | 含 B6 权重（142MB） |
| `AgroLabel_AL_full_dataset_v3.zip` | AL 效率消融：7773 图 + labels + `m8_al.py` + **官方冷启动权重** | 含 `yolo26x-seg.pt`，**不含 B6** |
| `exp60_arms_package_v2.zip` | 三臂人工效率对照：68 张 val 图 + GT + B6 预标注备份 + v1.1 协议 | 不含权重，依赖上面 B6 包 |

**红线**：AL 训练起点只能用官方 `yolo26x-seg.pt`，绝不能用 B6（B6 看过全部 5367 张 train，会造成数据泄露）。B6 只用于演示和人工编辑对照。

## 二、B6 包内容

| 文件 | 说明 |
|---|---|
| `b6_y26x_1280_best.pt` | 当前最优模型（YOLO26x-seg@1280，val mask mAP50 0.416 / 50-95 0.214），142MB，GPU 推理约 144ms/图 |
| `docs/INTERFACE.md`、`docs/INTERFACE_ADDENDUM.md` | 前后端接口契约（前端已在用，改动需同步） |
| `docs/模型实验作战手册.html` | 全部实验台账（哪些路线已测死、哪些有效） |
| `docs/al_effect_smoke.png` | 主动学习冒烟效果可视化 |
| `docs/THREE_ARMS_PROTOCOL.md` | 三臂实验 v1.1 协议（与 `exp60_arms_package_v2.zip` 一致） |
| `docs/HANDOFF_TEAMMATE.md` | 本说明 |

## 三、本地跑通三步（Windows）

```bash
# 1. 环境（Python 3.10+，需 NVIDIA GPU）
pip install fastapi uvicorn ultralytics opencv-python pyyaml pillow numpy

# 2. 数据集：用 AgroLabel_AL_full_dataset_v3.zip 解压后的目录
#    三臂实验只用它里面的 images/val，禁止使用后端 al_pool 那 36 张 train 图

# 3. 启动后端（在仓库根目录；数据/权重路径用环境变量切换，不改代码）
set AGROLABEL_DATA_ROOT=<AL_v3 解压目录>/dataset
set AGROLABEL_WEIGHTS=<解压后的 b6_y26x_1280_best.pt 绝对路径>
set AGROLABEL_IMGSZ=1280
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000

# 前端（Vite 默认只听 IPv6，必须加 --host）
cd frontend && npm install && npm run dev -- --host
# 打开 http://127.0.0.1:5173/
```

## 四、验证后端就绪

- `http://127.0.0.1:8000/api/health` → 应返回 `"model":"b6_y26x_1280_best.pt"`、`"imgsz":1280`；
- 页面打开任意 val 图 → 红色多边形掩码 + 右侧实例置信度，即 B6 预标注生效；
- `image_id` 不带扩展名，带 `.jpg` 会 404。

## 五、三臂实验口径（必须与 `exp60_arms_package_v2.zip` 一致）

| 臂 | 后端臂名 | 起点 | 操作 |
|---|---|---|---|
| A 纯人工 | `manual` | 空白画布 | 从零画多边形 + 选类 |
| B 预标注+编辑 | `assisted` | 实时 B6 预标注 | 拖边/删除/补漏/改类，点提示可用 |
| C 点提示消融 | `noprompt` | 同 B | **禁用点提示**，测交互式精修独立贡献 |

- 68 张 val 图，每图只进一个臂；分配为 manual 23 / assisted 23 / noprompt 22；
- B/C 预标注实时调 `/api/predict`；`predictions_b6.json` 仅作离线备份；
- 主计时用后端 `elapsed_from_open_ms`，`timing_sheet.csv` 仅备份；
- 汇总用 `python backend/tools/human_stats.py --with-editcost`；
- **AL 排序只做演示截图，不进三臂统计表**。

## 六、接口要点（踩过的坑，勿重复踩）

1. `image_id` 是文件 stem，不带 `.jpg`；
2. `/api/images` 分页参数是 `limit/offset`，单页上限 200，`page_size` 无效；
3. `/api/sample` 单次 pool 上限 500 张，全池 7773 需分批；
4. 前端 Vite 必须 `--host` 才能用 127.0.0.1 访问；
5. 预测延迟 ~120–260ms/图（GPU），低配机可 `AGROLABEL_DEVICE=cpu`（慢 10 倍以上）。

## 七、当前主线状态（避免重复劳动）

- **已测死**：SAM3 全参微调；零样本全家桶（YOLO-World / YOLOE / GDINO+SAM2，mAP≈0）；语义投票融合两代均阴性。
- **已收官阴性**：Mask2Former-v2 于 11-17 完成 50 epoch，segm mAP/mAP50 = 0.140/0.281，低于 B6；保留权重作证据，不再追加训练。
- **有效主线**：B6@1280 全监督 + AL 闭环。AL 用官方 `yolo26x-seg.pt` 冷启动，B6 只用于演示和人工编辑对照。
