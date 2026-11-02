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

当前状态：**46 项断言全部通过**。

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
| `POST /api/logs/ops` | ✅ | 追加 `out/logs/ops.csv`，表头对齐 `src/m4_app.py:131` |
| `GET /api/logs/export` | 🕒 | **P3 占位**：已注册路由，一律返回 501 + 说明（不是 404） |
| `POST /api/sample` | ✅ | 调模型侧 `select()`；默认剔除已标注图，避免反复推荐同一批 |
| `POST /api/refine` | ✅ | 调模型侧 `refine()`；点一下 ROI 重识别，实测 78 ms |
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

## 数据源：两种模式，自动切换

后端启动时按顺序探测：

1. 环境变量 `AGROLABEL_DATA_ROOT` 指向的目录
2. `configs/dataset_plantseg34.local.yaml` 里 `path` 声明的目录
3. 都不可用 → 退用仓库内 **`experiments/sam3_sample_vis/`**（102 张真实叶片图）

`/api/health` 的 `dataset.mode` 会明确告诉你现在是 `manifest` 还是 `demo`。

| 模式 | 索引来源 | 用途 |
|---|---|---|
| `manifest` | `datasets/sam3/plantseg34_{train,val}.json`（COCO），路径拼到数据集根 | 正式实验 |
| `demo` | 扫 `experiments/sam3_sample_vis/*.jpg` | 前端联调，**不能用来跑实验** |

> **为什么要有 demo 模式**：仓库里不带数据集（合规 + 体积），队友拉下代码如果必须先配好
> `F:/_plantseg` 才能看到界面，前端会被卡住。demo 模式让两边可以并行开工。

`manifest` 模式下，清单里声明但磁盘上不存在的条目会被**静默跳过**并计入 `skipped`，
启动日志会打印「索引 N 张，跳过 M 条」。本机用 2 张假图验证过这条路径。

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
    └── smoke_test.py       # 46 项断言，仅依赖标准库
```

---

## 与模型侧的关系

**本服务不 import `src/` 里的任何模块。** 它直接加载 `weights/` 下的 `.pt` 用 ultralytics 推理。

这样做是因为 `src/` 里的脚本（`m1_infer.py` 等）都是带 argparse 的 CLI，不是库函数，
与其包一层 subprocess（多一次进程启动 + 序列化开销，300 ms 预算扛不住），不如直接用同一套 ultralytics API。

等模型侧把 `inference/predict()` 和 `active_learning/select()` 做成可 import 的函数，
`model_bridge/predictor.py` 换实现即可，对外契约不变。
