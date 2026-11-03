# backend/ —— 标注服务

把模型能力包装成 HTTP API，供 `frontend/` 调用。

契约：[`../docs/INTERFACE.md`](../docs/INTERFACE.md) + [`../docs/INTERFACE_ADDENDUM.md`](../docs/INTERFACE_ADDENDUM.md)

---

## 快速开始

```powershell
# 在仓库根目录
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload

# Swagger
# http://127.0.0.1:8000/docs
```

环境搭建（含本机的 pip 镜像、OMP 坑、CUDA 版 torch）见 [`../docs/ENV_SETUP.md`](../docs/ENV_SETUP.md)。

冒烟测试（另开一个终端）：

```powershell
.\.venv\Scripts\python.exe backend\tests\smoke_test.py
```

当前状态：**106 项断言全部通过**。

---

## 已实现

| 接口 | 状态 | 说明 |
|---|---|---|
| `GET /api/health` | ✅ | 数据源、索引张数、设备、权重 sha256 |
| `GET /api/classes` | ✅ | 34 类字典，源为 `configs/dataset_plantseg34.local.yaml` |
| `GET /api/images` | ✅ | 分页 / `ids=` 批量 / 按 split、class 过滤 |
| `GET /api/images/{id}/file` | ✅ | ETag + 304 + Range + immutable 缓存 |
| `POST /api/predict` | ✅ | **真模型**，GPU 单张，实测 28–48 ms |
| `POST /api/annotations` | ✅ | 落 `out/annotations/<arm>/`，YOLO seg txt + 元数据 json |
| `GET /api/annotations/{id}` | ✅ | 读回已保存的标注；没标过返回 200 + `found:false` |
| `POST /api/logs/ops` | ✅ | 追加 `out/logs/ops.csv`，表头对齐 `src/m4_app.py:131` |
| `GET /api/logs/export` | 🕒 | **P3 占位**：已注册路由，一律返回 501 + 说明（不是 404） |
| `POST /api/sample` | ✅ | 调模型侧 `select()`；默认剔除已标注图，避免反复推荐同一批 |
| `POST /api/refine` | ✅ | 调模型侧 `refine()`；顶点上限 20 并剪掉 <2px 的边 |
| `POST /api/edit-cost` | ✅ | 调 `m6_editcost.edit_ops()`；多边形先栅格化成掩码 |

### 三个"接线"接口的两个注意点

**`/api/sample` 的 `reasons.score` 恒为 `null`。**
模型侧 `select()` 按签名只返回路径列表、不返回分值，`rank` 即选样顺序。
要拿真实不确定性分数，需要 `m8_al` 的 `score_*` 一并返回分值（模型侧改动）。

**`/api/refine` 的 `class_id` 是硬过滤，前端必须传。**
实测同一张甘蓝图，不传 `class_id` 会返回 **Wheat**（0.34）——放大到小 ROI 后模型
丢掉了宿主上下文；传了正确的 `Cabbage` 才回本类。前端点提示时用"当前选中实例的类
→ 图像 gt 类"依次兜底。

`/api/sample` 的非 random 策略会在**每次请求里重新加载一次模型**
（`select()` 内部 `YOLO(...)`），20 张池实测 865 ms。池子大时注意别在前端同步等待。

`/api/predict` 除了契约里的 `instances` + `latency_ms`，还多返回
`image_width` / `image_height` / `model_version`（向后兼容的新增字段）。

### 实验臂（三臂对照）

`out/annotations/<arm>/` 按臂分目录，三臂互不覆盖。白名单只有三个：

| arm | 前端行为 | 用途 |
|---|---|---|
| `assisted` | 预标注 + 可改可删 + 可点提示 + 可手画 | 完整方案 |
| `manual` | **不跑 `/api/predict`**，打开是空白图 | 基线（纯人工） |
| `noprompt` | 跑预标注，但**禁用点提示** | 归因：减去预标注的贡献 |

**臂名走白名单校验，未知臂名一律 422。** 原因：臂名拼错一个字母（多打一个空格、
首字母大写）就会静默多出一个目录，三臂对照的分母就错了，而且事后极难发现。

拆解逻辑（技术方案 6.4 要的是这个）：

```
assisted  vs manual   = 预标注 + 点提示 的总收益
noprompt  vs manual   = 仅预标注的收益
assisted  vs noprompt = 点提示的净收益
```

### `/api/refine` 的调参依据（别乱改，这是实测出来的）

模型侧 `refine()` 的默认值在这个数据集上**几乎不可用**。实测（4~5 张 demo 图，
每张 5 个已预测病斑质心 + 5 个空白点，判据是"返回的多边形是否真的包住点击点"）：

| conf | radius | class_id | 有用 | 返回但不是你点的 | 没认出来 |
|---|---|---|---|---|---|
| 0.25 | 48（模型侧默认） | 限定 | **12%** | 6% | **81%** |
| 0.25 | 96 | 限定 | 38% | 19% | 44% |
| 0.10 | 96 | 不限定 | 75% | 25% | 0% |
| **0.10** | **112（短边一半）** | **限定 → 不限定兜底** | **67%** | **0%** | 33% |

三个结论：

1. **ROI 太小是主因**。默认半径 `0.08·min(W,H)`，302×225 的图上只有 48px，
   即 96×96 的窗口被放大到 640（6.7 倍），病斑尺寸远超训练分布。
   现在改成 `clamp(0.5·min(W,H), 96, 512)`。
2. **类别硬过滤是次因**。放大后模型常把甘蓝判成小麦，硬过滤就把正确答案丢了。
   现在先按给定类别找，找不到就**不限定类别再找一次**，并返回 `class_mismatch=true`，
   前端提示"识别为 X，可在右栏改类"。
3. **不能只信 `found`**。`refine()` 的 `best_centroid` 兜底没有距离限制，
   不限定类别时空白点也有 85~95% 会返回一个**并不包含点击点**的多边形。
   现在后端强制校验包含关系，不包住就返回 `found=false`（空白点误报降到 **0%**）。

时延：中位 16 ms、P90 24 ms（失败时要多跑一遍，也才 ~30 ms）。

### 真实 100 张上的实测（2026-11-03）

| 接口 | 耗时 | 备注 |
|---|---|---|
| `/api/predict` | 59 ~ 132 ms | 132×102 到 4608×3072 都测了，接口预算 300 ms ✓ |
| `/api/refine` | 55 ~ 62 ms | 真实病斑上 found=True，19 顶点 |
| `/api/sample` random | 35 ms | |
| `/api/sample` coreset | 0.7 s | |
| `/api/sample` entropy | **2.3 s** | 100 张全扫一遍前向 |
| `/api/sample` tta | **63 s** ⚠️ | 每张 4 次前向 × 100 张 = 400 次 |

TTA 这个数字对"点一下按钮"的交互是不可接受的，见 `docs/BACKLOG.md` 的 AL-TTA-LATENCY。
还有一点：**顶点密度**。模型侧 `max_vertices` 默认 64，实测能吐出 40 个顶点、
相邻两点只隔 **1.1 px** —— 在画布上重叠成一团，既看不清边界也没法拖。

后端把默认降到 **20**（数据集真值平均 22 顶点/实例），再把间距 <2px 的顶点并掉
（保留 ≥3 个，且剪枝后重新校验"还包住点击点"，否则退回原多边形）。
实测同一批点：**顶点 24~40 → 7~19，最短边 1.1 px → ≥2.45 px**。

### 标注与日志的几处实现约定

- **实验臂** `arm` 走查询参数：`POST /api/annotations?arm=assisted`。
  缺省取环境变量 `AGROLABEL_ARM`，再缺省 `assisted`。
  三臂（`assisted` / `manual` / `noprompt`）各落一个目录，互不覆盖（ADDENDUM §4.3）。
- **`pred_snapshot` 不由前端传**：提交时从 `out/pred_cache/<image_id>.json` 读（§4.2）。
  缓存缺失时记为 `[]`、响应 `pred_cached: false`，**但提交照常成功**。
- **`no_target` 也落一个空 txt**：空文件才代表"标过了、确实没目标"，
  和"从没标过"（文件不存在）靠文件存在性区分（§4.1）。
- **几何校验**：顶点 < 3、坐标超出 `[0,1]`、`class_id` 越界一律 422 打回。
  宁可报错也不要写出坏标签去污染 m6 的统计。
- **`ops.csv` 用 `utf-8-sig`**：`m4_app.py` 写、`m5_logstats.py` 用 `utf-8-sig` 读，
  不带 BOM 时中文 `detail` 会乱码。
- **`op` 词表严格校验**：`改类别/删实例/标为无目标/整张确认/撤销/提交/新画/拖边界`，
  词表外的值 422 打回 —— 这张表是 `m5_logstats.py` 硬编码的，多加一个词就等于统计失真。

### 闭环已验证

用真实浏览器操作（删 1 个实例 → 整张确认 → 提交）跑通：

```
POST /api/annotations  200  -> out/annotations/assisted/00_cabbage....txt（23 行）
POST /api/logs/ops     200  -> out/logs/ops.csv（3 行，带 BOM）
python src/m5_logstats.py  -> 已提交 1 张，均值 1.00 次/张，时延 42ms（线 300ms）
```

注意 m5 只把 `改类别 / 删实例 / 标为无目标` 计入修改次数，
`整张确认` 与 `提交` 是中性 —— 上面那次的 1 次就来自 `删实例`。

---

## 数据源：三种模式，自动切换

后端启动时按顺序探测：

1. 环境变量 `AGROLABEL_DATA_ROOT` 指向的目录（有 `images/` + `data.yaml` 就按 YOLO 处理，否则按 COCO 清单）
2. `configs/dataset_plantseg34.local.yaml` 里 `path` 声明的目录（同上自动识别）
3. **`datasets/plantseg100_pack/`**（组内的 100 张 × 34 类采样包，YOLO-seg 结构）
4. 都不可用 → 退用 `out/demo/` 或 `experiments/sam3_sample_vis/`

`/api/health` 的 `dataset.mode` 会明确告诉你现在是 `manifest` / `yolo` / `demo`。

| 模式 | 索引来源 | 用途 |
|---|---|---|
| `manifest` | `datasets/sam3/plantseg34_{train,val}.json`（COCO），路径拼到数据集根 | 正式实验（原始大盘） |
| `yolo` | 扫 `images/<split>/*.jpg` + `labels/<split>/*.txt` + `data.yaml` | 正式实验（采样包） |
| `demo` | 扫 `out/demo/*.jpg` | 前端联调，**不能用来跑实验** |

> **为什么要有 demo 模式**：仓库里不带数据集（合规 + 体积），队友拉下代码如果必须先配好
> `F:/_plantseg` 才能看到界面，前端会被卡住。demo 模式让两边可以并行开工。

`manifest` 模式下，清单里声明但磁盘上不存在的条目会被**静默跳过**并计入 `skipped`，
启动日志会打印「索引 N 张，跳过 M 条」。本机用 2 张假图验证过这条路径。

`yolo` 模式下：
- 类别名优先取**包内 `data.yaml`**，取不到才用 `configs` 里的那份
- `image_id` 用文件名主干（它要变成 `out/annotations/<arm>/<id>.txt`，不能带路径分隔符），
  跨 split 重名时才加 `<split>_` 前缀
- `gt_class_id` 取该图标注文件**第一行**的 class_id —— 面板只显示一个"这张图属于哪类"，
  同宿主多病害时取首个实例足够，不猜"多数类"
- 宽高用 PIL 现读（100 张约 0.1 s），坏图计入 `skipped` 而不是让整个索引失败

### 一个已知假设

`manifest` 模式的宽高取自 COCO 清单，**不重新解码原图**（5367 张全解码会让启动慢几十秒）。
如果换了不同版本的同一份数据，清单里的宽高可能与文件实际尺寸不符。
`/api/predict` 返回的 `image_width` / `image_height` 是解码后的真值，可用来校准。

---

## 环境变量

| 变量 | 缺省 | 说明 |
|---|---|---|
| `AGROLABEL_DATA_ROOT` | 读 `configs` | 数据集根目录 |
| `AGROLABEL_WEIGHTS` | `weights/baseline_yolov8nseg_plantseg34_best.pt` | 权重 |
| `AGROLABEL_DEVICE` | `0` | ultralytics 的 device；无卡自动降 `cpu` |
| `AGROLABEL_CONF` | `0.25` | 置信度阈值 |
| `AGROLABEL_IMGSZ` | `640` | 推理尺寸 |

---

## 两个必须知道的实现约束

**1. 导入顺序不能动**

`app.py` 开头这几行是实测结论，不是风格：

```python
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import cv2                     # 必须早于 torch
```

顺序错了会直接崩：`OMP: Error #15: Initializing libiomp5md.dll ... already initialized`
（conda 的 MKL 与 CUDA 版 torch 各带一份 `libiomp5md.dll`）。

**2. GPU 上是 CUDA tensor**

`result.boxes.conf` / `.cls` 在 GPU 推理后是 tensor，`np.argmax()` 会抛
`can't convert cuda:0 device type tensor to numpy`。
取之前必须 `.cpu().numpy()`（见 `model_bridge/predictor.py`）。

---

## 目录

```
backend/
├── app.py                  # 入口：环境兜底 + lifespan 加载一次模型 + 挂路由
├── config.py               # 路径 / 数据源判定 / 设备 / 类别字典
├── schemas.py              # Pydantic 模型，字段名对齐契约
├── routers/
│   ├── system.py           # /api/health, /api/classes
│   ├── images.py           # /api/images, /api/images/{id}/file
│   └── predict.py          # /api/predict
├── model_bridge/
│   ├── catalog.py          # 图像索引（manifest / demo）
│   └── predictor.py        # YOLO 封装 + 多边形简化 + pred 缓存
└── tests/
    └── smoke_test.py       # 106 项断言，仅依赖标准库
```

---

## 与模型侧的关系

分两种：

**已经用上模型侧的可调用实现**（推荐，口径与模型侧一致）：

| 接口 | 调用的函数 |
|---|---|
| `POST /api/sample` | `active_learning/select.py::select()` |
| `POST /api/refine` | `inference/predict.py::refine()` |
| `POST /api/edit-cost` | `src/m6_editcost.py::edit_ops()` |

**`POST /api/predict` 仍是本服务自己的封装**（`model_bridge/predictor.py`）。

原因：`src/` 里的脚本（`m1_infer.py` 等）都是带 argparse 的 CLI，包一层 subprocess
多一次进程启动 + 序列化开销，300 ms 预算扛不住。而 `inference/predict.py` 出现得比
后端晚，当时先按同一套 ultralytics API 写了一份。

> ⚠️ **这是一笔待还的技术债**：现在有两套 predict 实现并存
> （本服务的 `predictor.py` 与模型侧的 `inference/predict.py`）。
> 模型侧将来加 HAFM / P2 分支时，本服务不会自动跟着变，
> 会出现"模型侧指标是新的、接口返回是旧的"。
> 还法：把 `predictor.py` 的推理部分换成调 `predict_response()`，
> 只保留 `pred_cache` 与 Catalog 这两块本服务独有的职责。
