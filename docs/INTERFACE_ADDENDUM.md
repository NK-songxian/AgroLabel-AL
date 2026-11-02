# 接口契约 · 补充条款（v1.2 · 已确认）

> **状态：团队已确认，本文件与 `docs/INTERFACE.md` 同等有效。**
> 原计划「确认后并入 INTERFACE.md 再删除本文件」调整为**两者并存**：
> `INTERFACE.md` 保留为精简契约（模型侧只读那份），本文保留设计依据、实测数据与决策记录，
> 便于答辩时说明"为什么这么定"。
>
> 提出人：前后端负责人　提出时间：2026-11-02　确认时间：2026-11-02
>
> 依据：目前 `INTERFACE.md` 的 4 个接口只覆盖了"模型能力封装"，没有覆盖"标注工作流"。
> P1 验收标准是「完成一次完整修正，日志可导出」，按原契约**走不通**。本文补的是这一段。
>
> 术语：本文用 `§2.x` / `§3.x` / `§4.x` 指本文件自己的小节；
> 写到数据格式约定时一律写成 `INTERFACE.md §1`，避免和本文的 "## 1" 混淆。

---

## 1. 为什么必须补（一句话版）

现有 4 个接口能回答"这张图模型看到了什么"，但回答不了：

- **标哪张**（image_id 从哪来）
- **怎么显示**（图像像素从哪取）
- **改完存哪**（修正结果落盘）
- **日志怎么算数**（`m5_logstats.py` 读 `out/logs/ops.csv`，现在没有任何接口往那写）
- **漏检怎么补**（模型 Mask Recall 只有 0.32，一张图平均 7.2 个病斑要补 5 个，逐点描不可行）

而 §4 的三处口径问题，会导致就算接口写了、"修改次数"仍然算不准。

---

## 2. 新增接口（6 个 P1 必需 + 1 个建议 + 1 个可选）

**实现状态**（2026-11-02）：

| 接口 | 状态 | 备注 |
|---|---|---|
| `GET /api/images` | ✅ | `order` 只支持 `path`/`class` |
| `GET /api/images/{id}/file` | ✅ | ETag + 304 + Range(206) 都验过 |
| `GET /api/classes` | ✅ | 源为 `configs` 的 `names` |
| `POST /api/annotations` | ✅ | 含几何/类别校验，坏标签一律 422 |
| `POST /api/logs/ops` | ✅ | `utf-8-sig`，表头逐字对齐 m4_app |
| `POST /api/refine` | ✅ | 调模型侧 `refine()`；`class_id` 硬过滤 |
| `GET /api/annotations/{id}` | ✅ | 续标/刷新恢复；没标过返回 404 |
| `GET /api/logs/export` | 🕒 | 已注册路由，一律 501（P3） |

### 2.1 `GET /api/images` —— 标注队列 / 元信息批量查询

```
GET /api/images?split=train&offset=0&limit=50
GET /api/images?ids=0001,0002,0003          ← 用于 /api/sample 返回后的落地
```

| 参数 | 类型 | 说明 |
|---|---|---|
| `split` | str | `train` / `val` / `test` / `all`，默认 `all` |
| `offset` / `limit` | int | 分页，`limit` 上限 200 |
| `ids` | str | 逗号分隔，**给了就忽略 offset/limit**，最多 200 个 |
| `class_id` | int | 按真值类别筛 |
| `order` | str | `path`（默认）/ `class`。**`score_desc` 暂不支持**（需要全量的预标注分数索引，现在只有 predict 过的图有），传了会 400 |

```json
{
  "total": 2336,
  "items": [
    {
      "image_id": "0001",
      "file_name": "0001.jpg",
      "split": "train",
      "width": 1208,
      "height": 300,
      "gt_class_id": 0,
      "gt_class_name": "Apple",
      "has_annotation": false,
      "pred_score": null
    }
  ]
}
```

> `width/height` 必须给：前端要把归一化多边形还原成像素坐标，`<img>` 还没 load 完时就需要这个数。

> 另：响应里还带了 `disease` 字段（取自 COCO 清单的 `disease`，如 `apple scab`）。
> 它**不属于 34 类**，只是给人看的辅助信息，别拿去当类别用。

### 2.2 `GET /api/images/{image_id}/file` —— 图像像素

- 返回二进制，`Content-Type: image/jpeg`（或 png）
- **必须**支持 `ETag` + `304`；**建议**支持 `Range`（大图/断点）
- `Cache-Control: public, max-age=31536000, immutable`（图片内容不会变）

> 不要用 base64 随 `/api/predict` 返回：一张 1600×1200 就 1.5 MB 字符串，
> 300 ms 预算里光编解码就吃掉一大块，而且会让浏览器无法复用缓存。

### 2.3 `GET /api/classes` —— 类别字典

```json
{
  "nc": 34,
  "classes": [
    { "class_id": 0, "class_name": "Apple" },
    { "class_id": 1, "class_name": "Banana" }
  ]
}
```

> 数据源固定为 `configs/dataset.yaml` 的 `names`，**不要在前端另写一份**。
> 用途：改类别下拉框、图例、配色映射。

### 2.4 `POST /api/annotations` —— 提交修正结果

**这是整个闭环的落点。** 请求体：

```json
{
  "image_id": "0007",
  "label_state": "labeled",
  "instances": [
    { "class_id": 0, "score": null, "polygon": [[0.075,0.39],[0.073,0.396]], "bbox": [0.067,0.39,0.095,0.579] }
  ],
  "elapsed_ms": 18420,
  "operator": "u01",
  "session_id": "s20261102-01"
}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `image_id` | ✅ | |
| `label_state` | ✅ | `labeled` / `no_target`，见 §4.1 |
| `instances` | ✅ | 最终结果；`no_target` 时传 `[]` |
| `elapsed_ms` | ✅ | 本张图的累计操作时长（前端埋点算） |
| `operator` / `session_id` | 建议 | 多人对照实验时要能区分 |
| `pred_snapshot` | **不传** | 由后端缓存填充（见 §4.2）。客户端若仍传，以缓存为准并忽略 |

响应：

```json
{ "ok": true, "image_id": "0007", "n_instances": 1, "pred_cached": true, "saved_at": "2026-11-02T14:31:05+08:00" }
```

> `pred_cached` 告诉调用方"这张图有没有关联到模型预标注"，便于压测时发现缓存没命中。
> `manual` 臂（纯人工，从未调过 `/api/predict`）时它应为 `false`，属正常。

**落盘位置**（重要，见 §4.3 的说明）：

```
out/pred_cache/<image_id>.json           # /api/predict 时写入，/api/annotations 时读取（见 §4.2）
out/annotations/<arm>/<image_id>.txt     # YOLO seg: class_id x1 y1 x2 y2 ...（归一化）
out/annotations/<arm>/<image_id>.json    # {label_state, pred_snapshot, elapsed_ms, operator, ts}
```

- `<arm>` = 实验臂，取值 `assisted`（预标+修正）/ `manual`（纯人工对照）/ `noprompt`（可选第三臂，见技术方案 6.4）
- `out/` 已在 `.gitignore` 中，不会误提交
- **不要写回 `datasets/`** —— 原始数据只读（交接文档明令禁止碰）
- ⚠️ `m6_editcost.py:216` 的 `images_to_labels()` 目前假定 `images/xxx/a.jpg → labels/xxx/a.txt`。
  沿用它，跑 m6 前需要把 `out/annotations/<arm>/` 铺成 `labels/` 结构，
  或给 m6 加一个 `--labels-dir` 参数（约 5 行）。**这是需要确认的一项，见 §7 第 2 条**。

### 2.5 `POST /api/logs/ops` —— 上报操作日志

```json
{
  "session_id": "s20261102-01",
  "events": [
    { "ts": "2026-11-02 14:30:01", "image": "0007", "op": "改类别",
      "instance": 0, "detail": "Apple -> Tomato", "n_ops_on_image": 1, "api_ms": 12.4 }
  ]
}
```

响应：`{ "ok": true, "appended": 1, "path": "out/logs/ops.csv" }`

**CSV 表头必须与 `src/m4_app.py:131` 完全一致，一个字段都不能多、不能少：**

```
ts, image, op, instance, detail, n_ops_on_image, api_ms
```

**编码必须 `utf-8-sig`（带 BOM）** —— `m4_app.py` 写、`m5_logstats.py` 用 `utf-8-sig` 读，
换成无 BOM 的 UTF-8 会让带中文的 `detail` 在某些环境下读成乱码。

**`op` 词表必须落在这个集合里**（`m5_logstats.py` 是硬编码的）：

| 归类 | 取值 | m5 是否计入修改次数 |
|---|---|---|
| 计数 | `改类别` `删实例` `标为无目标` | ✅ 各 +1 |
| 中性 | `整张确认` `撤销` `提交` | ❌ |
| 扩展 | `新画` `拖边界` | ❌（m6 从几何算，不进 m5） |

> `image` 字段填**文件名的 stem**（无扩展名），不是 image_id、也不是全路径 —— `m4_app.py` 就是这么写的，
> `m5_logstats.py` 按它分组。

### 2.6 `POST /api/refine` —— 点提示：裁 ROI 放大重识别（**P1 必需**）

> **为什么需要它**：分发的基线权重 Mask Recall 只有 **0.321**、每图平均 **7.2** 个病斑，
> 意味着每张图要人补 ~5 个。而病斑面积中位只占整图 **0.3%**，逐点描一个多边形要 22 次点击。
> 实测：把图切开送模型，检出数提升 **4.5 倍**（3×3），且多出来的置信度分布与整图一致
> —— 说明这些病斑不是"模型不会"，是"太小看不见"。
> 所以补漏检的正确姿势是**点一下，让模型在放大后的局部再看一眼**，而不是人手描。
> 实测单次耗时 **8.8 ms**（测的是 320×320 crop 送 640，GPU；默认半径算出的 crop 更小，只会更快）。

> ⚠️ **2026-11-02 实测修正**：上面那句"crop 更小只会更快"掩盖了一个真问题 ——
> **crop 太小会认不出**。模型侧默认半径 `0.08·min(W,H)` 在 302×225 的图上只有 48px，
> 96×96 被放大到 640 是 6.7 倍，病斑尺寸远超训练分布。实测该默认值下只有 **12%**
> 的点击能找到东西，**81% 都是"没认出来"**。
>
> 后端已改为：半径 `clamp(0.5·min(W,H), 96, 512)`、局部 conf 降到 0.10、
> 类别先限定后兜底、并**强制校验返回的多边形是否包住点击点**。
> 实测 **67%** 能找到，空白点误报从 85~95% 降到 **0%**。详见 `backend/README.md`。
>
> 响应因此多了两个字段：`class_mismatch`（识别类别与请求不一致，前端提示可改类）、
> `radius_used`（排查"为什么没认出来"用）。

**请求**

```json
{
  "image_id": "0007",
  "point": [612, 184],
  "radius": 96,
  "class_id": 0,
  "max_vertices": 64
}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `image_id` | ✅ | |
| `point` | ✅ | `[x, y]`，**原图像素坐标**（点击天然是像素，不归一化） |
| `radius` | — | ROI 半边长（像素）。**省略时**按 `clamp(round(0.08 × min(W,H)), 48, 512)` 计算；显式传则 `clamp` 到 `[48, min(1024, ⌊min(W,H)/2⌋)]` |
| `class_id` | — | 已知类别时传。传了就只在该类里找，避免同类病斑串类 |
| `max_vertices` | — | 返回多边形的**顶点上限**，默认 64，范围 16–256 |

> **`radius` 为什么不能写死**：实测图像短边 P50 只有 **500 px**（P10 270 / P90 1152 / max 4000），
> 病斑等效直径 P50 **37 px**。固定 160 px 半径在 400×300 的小图上会把整张图圈进去。
> 前端也可以按当前视口自己算，推荐 `radius = clamp(0.5 × min(视口宽,视口高) / 当前缩放, 48, 512)`
> —— 这样"放大后再点"天然得到更小的窗，行为可预测。

**响应**

```json
{
  "found": true,
  "class_id": 0,
  "class_name": "Apple",
  "score": 0.5834,
  "polygon": [[0.0751, 0.3899], [0.0734, 0.3962]],
  "bbox": [0.0672, 0.3899, 0.0953, 0.5786],
  "coverage": 0.21,
  "raw_vertices": 317,
  "latency_ms": 8.8
}
```

| 字段 | 说明 |
|---|---|
| `found` | **必须显式返回**。点空处/ROI 内无目标时为 `false` |
| `polygon` | **归一化**（与 `INTERFACE.md` §1 的实例约定一致），前端 `× width/height` 还原。**顶点数 ≤ `max_vertices`** |
| `bbox` | 归一化 `[xmin, ymin, xmax, ymax]` |
| `coverage` | 该实例掩码面积 / ROI 面积。**> 0.85 时前端提示"模型把整个选区当成一个实例了，请缩小半径或换位置"** |
| `raw_vertices` | 简化前的原始顶点数，仅用于排查 |

#### ⚠️ 必须做多边形简化

实测：同样一个病斑，**整图预测返回 32 个顶点，裁 ROI 放大后返回 317～895 个顶点**（10 个样本中位 317，最大 895）。
原因很简单——放大后轮廓的分辨率变高了。

后果很严重：

- 900 个顶点的多边形在 Konva 上**没法拖**（用户要改边界，面对 900 个顶点无从下手）
- 单个实例 JSON 就有 ~18 KB，一次返回多个会明显拖慢
- 与真值分布严重不符（GT 顶点数中位 **22**、P90 **34**、最大 64）

**所以后端必须在返回前简化**，建议直接用 OpenCV：

```python
# eps 按周长比例取，保证顶点数跟尺度无关
peri = cv2.arcLength(contour, True)
eps  = max(0.5, 0.004 * peri)
approx = cv2.approxPolyDP(contour, eps, True)
# 若顶点数仍超 max_vertices，逐步加大 eps 重试
```

目标：**顶点数落在 16～64**，与 GT 分布对齐。前端拖 20 个顶点是可行的，拖 900 个是不可能的。

**没检出时**（这是常见情况，不是错误）：

```json
{ "found": false, "polygon": null, "bbox": null, "latency_ms": 7.9 }
```

> **不要返回 404 / 空数组 / 抛异常** —— 前端要靠 `found:false` 走"请手动框选"兜底。
> 这是 restful 意义上的"查询成功但无结果"，不是服务端错误。

**坐标换算（后端实现）**

```
r  = radius（已 clamp）
x0 = max(0, min(point.x - r, W - 2r))      # 贴边时把窗口整体向内平移，
x1 = min(W, x0 + 2r)                        # 保证 crop 仍是完整的 2r 宽、且一定含点击点
y0 = max(0, min(point.y - r, H - 2r))
y1 = min(H, y0 + 2r)

crop    = img[y0:y1, x0:x1]
result  = predict(crop)                     # 复用模型侧 predict(image_path)
选中实例 = 包含 point 的实例中 score 最高者；
          若无实例包含 point，则取质心离 point 最近的
polygon_norm[i] = [ (x0 + px) / W , (y0 + py) / H ]
coverage        = mask_area / ((x1 - x0) * (y1 - y0))
```

> **别写成 `x0 = clamp(point.x - r, 0, W)`** —— 那样贴边时 `x0 + 2r` 会超出 `W`，
> numpy 切片虽然会自动截断，但 `coverage` 的分母就成了错的，前端"框太大"的判断跟着失效。
> 用上面"向内平移"的写法，crop 尺寸恒为 `2r × 2r`（图像本身比 `2r` 小时除外）。

**侧别约定**

- **输入像素、输出归一化** —— 这个不对称是故意的：点击天然是像素，而系统里所有 `polygon` 统一归一化。
  **实现时务必按上面两行公式换算，别把两种坐标混用。**
- 纯读接口，无副作用，**可安全重试**。

**性能预算**

| | 实测 |
|---|---|
| 单次 ROI 重识别（GPU） | **8.8 ms** |
| 契约预算 | ≤ 100 ms（交互级，用户点完要立刻出结果） |

**模型侧影响**：**零**。后端只需把 crop 落成临时文件，调现有的 `predict(image_path)`，不新增模型能力、不需要 SAM。

**后端实现的两个已知雷**（实测踩过，写在这里省得再踩）：

1. **GPU 上 ultralytics 返回的是 CUDA tensor**，`np.argmax(result.boxes.conf)` 会直接抛
   `TypeError: can't convert cuda:0 device type tensor to numpy`。
   取任何 `conf` / `cls` / `masks` 之前先 `.cpu().numpy()`。
2. **`result.masks.xy` 给的是相对 crop 的坐标**，必须加上 `crop_x0/crop_y0` 再除以原图
   `W/H` 才是归一化坐标。少了这一步，多边形会整体偏移一个 ROI 的距离。

**降级路径**：`found:false` 或 `coverage > 0.85` 时，前端切到"手画矩形 / 多边形"模式。
拖顶点微调是纯前端几何操作，**不需要后端接口**。

### 2.7 `GET /api/annotations/{image_id}` —— 读回（续标用，建议做）

```json
{ "image_id": "0007", "label_state": "labeled", "instances": [...], "elapsed_ms": 18420 }
```

未标注时返回 `{"image_id":"0007","label_state":"unlabeled","instances":[]}`，**不要返回 404**。

### 2.8 `GET /api/logs/export?format=csv` —— 导出（可选，P3）

直接返回 `out/logs/ops.csv` 文件流。用途：技术报告附录、答辩现场展示原始证据。

---

## 3. 对现有 4 个接口的扩展（**向后兼容，不破坏已有实现**）

### 3.1 `/api/predict`

| 变更 | 内容 |
|---|---|
| 入参收敛 | `image_id` **优先**；`image_base64` 保留但注明"仅 ≤512 KB 小图，服务端需额外解码，会占用 300 ms 预算" |
| 响应**新增** `image_width` / `image_height` | 前端还原归一化多边形必需 |
| 响应**新增** `model_version` | 写进操作日志，出问题能追溯到哪版权重 |

```json
{
  "instances": [{ "class_id": 0, "class_name": "Apple", "score": 0.5801,
                  "polygon": [[0.075,0.39]], "bbox": [0.067,0.39,0.095,0.579] }],
  "latency_ms": 19.3,
  "image_width": 1208,
  "image_height": 300,
  "model_version": "baseline_yolov8nseg_plantseg34_best.pt@sha256:9f2c41ab"
}
```

> `model_version` 建议格式 `文件名@sha256前8位`，权重换了就能一眼看出来。
> 别塞 git commit —— 权重文件和代码提交不是一一对应的。

### 3.2 `/api/sample`

`reasons` 目前**没有定义结构**，前端无法展示"为什么选它"。补：

```json
{
  "selected": ["0007", "0012"],
  "reasons": {
    "0007": { "score": 0.91, "criterion": "entropy" },
    "0012": { "score": 0.77, "criterion": "coreset-distance" }
  }
}
```

### 3.3 `/api/edit-cost`

明确 `pred_instances` / `gt_instances` 复用 `INTERFACE.md` §1 的实例结构，且**必须含 `class_id` 与 `polygon`**（`bbox` 可选，服务端可从 polygon 推）。
判定口径写给实现方，避免两个人数出两个结果：

```
同类内贪心 IoU 匹配，IOU_MATCH = 0.5，IOU_FREE = 0.75
  gt 无匹配          -> 新画       +1
  pred 无匹配        -> 删实例     +1
  配对但 class 不同   -> 改类别     +1
  配对且 IoU < 0.75  -> 拖边界     +1
  配对且 IoU >= 0.75 -> 不计数
  gt 与 pred 均空     -> 标为无目标 +1
其余情况              -> 0
```

> 与 `src/m6_editcost.py` 的 `IOU_MATCH` / `IOU_FREE` 常量一致，改这里必须同步改那里。

---

## 4. 三处口径问题（不补的话，接口写了也算不准）

### 4.1 `label_state`：区分"没叶子"和"没标完"

YOLO seg txt 里两者都是空文件，**无法区分**。
`m5_logstats.py` 的统计分母是"已提交的图"，分不清就会把半成品算进分母，提效比全歪。

| 取值 | 含义 | 落盘 |
|---|---|---|
| `labeled` | 已标完（可能有实例，也可能人工判定无目标但已确认） | `.txt` 有内容或空 |
| `no_target` | 确认这张图没有目标 | `.txt` 空 + `.json` 记 `no_target` |
| `unlabeled` | 还没标（**仅出现在查询响应里，不应作为提交值**） | 无文件 |

### 4.2 `pred_snapshot`：由后端缓存，让"修改次数"事后可复算

**已决定：前端不传，后端缓存。** 理由是前端回传一份可能几 MB 的数组既浪费又容易漏。

**问题背景**：`m6_editcost.py` 算的是 `pred vs gt` 的几何差异。
如果只存最终结果，"提交前模型给的那一版"就永久丢失，事后无法离线复算编辑代价，
只能完全依赖前端埋点 —— 埋点漏记一次，那个"下降 30%"就再也对不上账。

**实现要求（后端）**

1. `/api/predict` 成功返回时，把**本次返回的 `instances` 原样**写一份到
   `out/pred_cache/<image_id>.json`（含 `model_version` 与时间戳）。
2. `/api/annotations` 提交时，读这个文件，填进 `out/annotations/<arm>/<image_id>.json` 的 `pred_snapshot`。
3. **必须落盘而不是放内存**：进程重启、换端口调试都不会丢；出了问题也能直接打开看。

**覆盖与缺失策略**

| 情况 | 行为 |
|---|---|
| 同一张图多次调 `/api/predict` | **后写覆盖先写**（用户可能调了两次阈值），记 `pred_ts` 便于排查 |
| 从未调过 `/api/predict`（`manual` 臂） | `pred_snapshot: []`，响应里 `pred_cached: false`，**属正常，不是错误** |
| `out/pred_cache/` 被清空 | 同上，退化成"无法复算"，但提交不能失败 |

> ⚠️ 上表最后一行是关键：**缓存缺失绝不能让提交 500**。标注结果是主数据，`pred_snapshot` 只是旁证。
> 丢了旁证可以接受，丢了一条标注不行。

### 4.3 为什么标注必须分 `<arm>` 目录，不能写进 `datasets/labels/`

技术方案 6.4 要做三臂对照（纯人工 / 预标修正 / 预标无点提示），每臂都要独立标一遍同一批图。
如果统一写进 `datasets/**/labels/`，后一臂会**覆盖**前一臂的结果，对照实验直接作废。

另外 `datasets/` 是原始数据区（企业脱敏样本），命题文件明确要求"数据合规、不外传"，
标注产物混进去既不好审计，也不好清理。

所以：**`out/annotations/<arm>/` 是唯一正确的落点。**

---

## 5. 标注交互流程（前后端共同依据）

四段式：**预标打底 → 人做"删/改类" → 漏检点一下放大重识别 → 手画兜底**。
这一节把每一步对应到哪个接口，避免两边各写一套。

| 步骤 | 用户动作 | 前端行为 | 调用 |
|---|---|---|---|
| 0 | 打开一张图 | 建 `session_id`，记 `image_open` | `GET /api/images`（队列）+ `GET /api/images/{id}/file` |
| 1 | 等预标注 | 整图推理 | `POST /api/predict` |
| 2 | 看一眼，没问题 | 记 `整张确认` | — |
| 3 | 某个实例类别错了 | 下拉改类，记 `改类别` | — （纯前端状态） |
| 4 | 某个实例是误检 | 删掉，记 `删实例` | — （纯前端状态） |
| 5 | **发现漏检** | 在病斑上点一下 → 转圈 → 出多边形 | **`POST /api/refine`** |
| 6 | 第 5 步没认出来 / 框太大 | 切到手画：矩形拖一下，或多边形点点 | — |
| 7 | 拖顶点微调边界 | 纯几何操作 | — （纯前端） |
| 8 | 这张图没目标 | 点"标为无目标"，记 `标为无目标` | — |
| 9 | 提交 | 只传最终 `instances`（`pred_snapshot` 由后端补） | `POST /api/annotations` |
| 10 | 连续标注 | 每步操作异步攒批上报 | `POST /api/logs/ops` |
| 11 | 拉下一批 | 主动学习选样 | `POST /api/sample` → 再用 `GET /api/images?ids=` 落地 |

**三条实现约定：**

1. **第 3/4 步不调后端。** 改类、删除、拖顶点都是前端状态变更，只有第 9 步提交才落盘。
   这样单张图 5 次修改 = 5 次本地操作 + 1 次提交，不会把网络往返摊进 300 ms 预算。
2. **`pred_snapshot` 由后端在第 1 步落盘（§4.2），前端不管。**
   后端必须保证它存的是**第 1 步的原始输出**，不能被后续请求覆盖成改过的版本 ——
   否则 `m6_editcost.py` 算出来的编辑代价恒为 0。
3. **第 5 步失败是正常路径**（`found:false`），必须走第 6 步兜底，不能弹错误框。

**编辑次数的口径**（与 `m5_logstats.py` 一致，前端埋点必须照这个记）：

```
计入：改类别 / 删实例 / 标为无目标      → 各 +1
不计：整张确认 / 撤销 / 提交 / 新画 / 拖边界
```

> 注意 `新画`（第 6 步手画）在 m5 里**不计入修改次数**，但建议仍然记日志，
> 因为它要进 m6 的 breakdown，也是技术报告里"模型漏检率"的直接证据。

---

## 6. 兼容性说明

| 对象 | 影响 |
|---|---|
| 模型侧 `src/*.py` | **零影响**。本文不改 `predict()` / `select()` 签名，也不动 `m4/m5/m6` 的任何常量 |
| 现有 4 个接口 | **向后兼容**。只新增响应字段，不删不改已有字段 |
| 后端已有 mock | 需补 **8 个路由**（6 个 P1 必需 + 1 建议 + 1 可选）；其中 7 个是薄封装，`/api/refine` 稍重（要裁图 + 简化多边形） |
| 前端 | 这些都是它本来就需要的数据源，不补才需要绕路 |

`/api/refine` 属后端自身封装（把 crop 落临时文件再调现有 `predict(image_path)`），
**不涉及模型侧任何改动**，也不需要 SAM / 新权重。

---

## 7. 决议记录（2026-11-02 团队确认，全部通过）

| # | 事项 | 决议 | 落地要求 |
|---|---|---|---|
| — | `pred_snapshot` 归属 | **后端缓存** | 前端不传；缓存 `out/pred_cache/<image_id>.json`；缺失时 `[]` 且提交不得失败（§4.2） |
| 1 | `label_state` 三取值 | **认可** | `labeled` / `no_target` / `unlabeled`（§4.1） |
| 2 | 标注落盘位置 | **认可 `out/annotations/<arm>/`**；`m6_editcost.py` 的 `--labels-dir` **由后端加** | 后端负责让 m6 能读到，不改 m6 的既有行为 |
| 3 | `ops.csv` 的 `image` 字段 | **用文件名 stem** | 与 `m4_app.py` 一致，`m5_logstats.py` 不动 |
| 4 | `/api/logs/export` | **留位置，P3 再做** | 路由先注册好返回 501 + 说明，不要 404 |
| 5 | `/api/refine` 输入像素 / 输出归一化 | **接受** | 后端换算，前端别混用（§2.6 有公式） |
| 6 | 多边形简化 ≤64 顶点 | **接受** | `cv2.approxPolyDP`，eps 取周长 0.004，不足再放大 |
| 7 | `/api/refine` 排期 | **同意 P1 末尾合入** | 不阻塞主链路 |

> 以上决议已同步到 `backend/README.md` 的实现清单与前端交互设计。
> 后续如需变更，仍按 RULE 01：先发群里确认，再改本文档。
