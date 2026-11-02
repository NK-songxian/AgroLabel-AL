# frontend/ —— 标注工作台

React 19 + TypeScript 7 + Vite 8 + react-konva 19 + Zustand 5 + Ant Design 6。

契约见 [`../docs/INTERFACE.md`](../docs/INTERFACE.md) 与 [`../docs/INTERFACE_ADDENDUM.md`](../docs/INTERFACE_ADDENDUM.md)。

![界面截图](../docs/screenshot-ui.jpg)

> 截图是真实渲染的：后端真模型在 302×225 的甘蓝黑腐病图上预标出 24 个病斑，
> 每个多边形都压在病斑上；右栏是实例列表与 34 类可搜索下拉。

---

## 跑起来

先起后端（另开一个终端，在仓库根目录）：

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

再起前端：

```powershell
cd frontend
npm install          # 首次
npm run dev          # http://localhost:5173
```

`vite.config.ts` 已把 `/api` 代理到 `127.0.0.1:8000`，所以前端代码里一律用相对路径；
生产部署时把 `dist/` 挂到反代后面即可，不用改任何 base 配置。

换后端地址：`$env:VITE_API_TARGET='http://127.0.0.1:8001'; npm run dev`

---

## 交互（与契约 §5 一一对应）

四段式：**模型预标打底 → 人做"删 / 改类" → 漏检点一下让模型放大重识别 → 手画兜底**。

| 操作 | 怎么做 | 走哪个接口 |
|---|---|---|
| 选图 | 左栏点任意一行 | `GET /api/images` + `GET /api/images/{id}/file` |
| 预标注 | 打开即自动跑 | `POST /api/predict` |
| 改类别 | 右栏实例行的下拉框（可搜索） | 纯前端 |
| 删实例 | 选中后按 `Del`，或右栏"删除选中" | 纯前端 |
| **补漏检** | 切到 **点提示** 模式后在病斑上点一下 | **`POST /api/refine`** |
| **智能选样** | 左栏选策略 + 点"智能选样" | **`POST /api/sample`** |
| 整张确认 | 右栏按钮 / `Enter` | 纯前端 |
| 标为无目标 | 右栏按钮 | 纯前端 |
| 撤销 | 右栏按钮 / `Ctrl+Z` | 纯前端 |
| 提交 | 右栏"提交并下一张" | `POST /api/annotations` |

**改类、删除、拖顶点都不调后端**，只有提交才落盘 —— 这样一张图改 5 次 = 5 次本地操作 + 1 次请求，
不会把网络往返摊进 300 ms 预算（`INTERFACE_ADDENDUM.md` §5 约定 1）。

### 快捷键

| 键 | 作用 |
|---|---|
| `←` `→` | 上一张 / 下一张 |
| `Del` / `Backspace` | 删除选中实例 |
| `Enter` | 整张确认 |
| `Esc` | 取消选中 |
| `Ctrl+Z` | 撤销 |
| `F` | 自适应视图 |
| `V` | 选择模式 |
| `P` | 点提示模式 |

鼠标：滚轮缩放（以光标为锚点）、拖拽平移、双击自适应、点多边形选中。

### 点提示（补漏检）

基线模型 Mask Recall 只有 0.32，一张图平均 7.2 个病斑要人补 5 个；
而病斑面积中位只占整图 0.3%，逐点描一个多边形要 22 次点击。

所以画布左上有个 **点提示** 模式（`P`）：在漏检的病斑上点一下，
前端按**图像像素坐标**调 `/api/refine`，后端裁 ROI 放大重识别，8.8~78 ms 返回多边形，
直接插入成新实例并记一条 `新画` 日志。

两条实现约定：

1. **一定要传 `class_id`**。它是硬过滤。实测不传的话，同一张甘蓝图会返回 `Wheat`（0.34）——
   放大到小 ROI 后模型丢掉了宿主上下文。前端用"当前选中实例的类 → 图像 gt 类"依次兜底。
2. **`found=false` 是正常路径**，不是错误。点healthy叶片、点背景都会返回它，
   此时提示"这个位置没认出来，请改用手画"，走手画兜底。

### 智能选样

左栏选策略 → 点"智能选样" → `/api/sample` 返回 id 列表 → 再 `GET /api/images?ids=`
取元信息 → 整队换成这批被选中的图，标题变成「选样 · entropy」。

后端默认剔除**已标注过**的图（扫 `out/annotations/*/*.txt`），所以反复点不会一直推荐同一批。

> `reasons.score` 恒为 `null`：模型侧 `select()` 只返回路径不返回分值。
> `rank` 是选样顺序，可以用来展示"优先标哪个"。

---

## 操作日志

右侧底部两个按钮：

- **下载 ops.csv** —— 生成 `ts,image,op,instance,detail,n_ops_on_image,api_ms` 表头的 CSV
- **复制 CSV** —— 同上，进剪贴板

表头与 `src/m4_app.py:131` **逐字一致**，`src/m5_logstats.py` 可直接 `csv.DictReader` 读。
`op` 词表严格限定在 `改类别` / `删实例` / `标为无目标` / `整张确认` / `撤销` / `提交` / `新画` / `拖边界`。

### 自动上送

日志**同时**走两条路：

1. **实时上送**：每攒够 5 条、或停手 0.8 秒，就 `POST /api/logs/ops`，由后端追加进
   `out/logs/ops.csv`。这样 `src/m5_logstats.py` 随时能算出真实数字。
2. **本地兜底**：内存里的 `ops` 数组始终完整，后面两个按钮随时可导出。

上送是**不阻塞标注**的：失败就把事件留在队列里等下次，连续失败 3 次自动停用自动上报
（避免后端没起来时每个操作都打一次无效请求），此时本地 CSV 导出仍然可用。

> 提交时会**强制冲一次队列再翻页** —— `m5_logstats.py` 靠 `提交` 事件确定分母，
> 少这一条它就会报"还没有已提交的图，无法统计"。

---

## 目录

```
src/
├── main.tsx                    # 挂载 + antd 深色主题
├── App.tsx                     # 三栏布局 + 启动屏 + toast
├── api.ts                      # 带类型的 API 客户端
├── store.ts                    # Zustand：队列 / 当前图 / 实例 / 撤销 / 日志
├── types.ts                    # 与契约一一对应的类型
├── colors.ts                   # 34 类黄金角配色
├── styles.css                  # 除 antd Select 外全部手写
├── hooks/useHotkeys.ts
└── components/
    ├── TopBar.tsx              # 数据集模式 / 设备 / 时延 / 翻页 / 缩放
    ├── QueuePanel.tsx          # 左栏队列
    ├── CanvasStage.tsx         # Konva 画布：图片 + 多边形 + 顶点
    └── InspectorPanel.tsx      # 右栏实例列表 + 动作 + 统计 + 导出
```

---

## 关于参考 X-AnyLabeling

可以，而且是**应该**参考的 —— 但要分清哪些抄、哪些不抄。

**抄这三样：**

1. **数据格式**（`anylabeling/views/labeling/label_file.py` + `shape.py`）。
   `LabelFile` 的 JSON 是事实标准：
   `{version, flags, shapes[], imagePath, imageData, imageHeight, imageWidth}`，
   每个 shape 是 `{label, score, points, group_id, difficult, shape_type, flags, description, attributes, kie_linking}`。
   我们内部用归一化坐标，但**导入导出时按它的结构序列化** ——
   这样队友拿桌面版 X-AnyLabeling 人工标的数据能直接进流水线，交付时也好看。
2. **交互约定**：`Ctrl+Z` 撤销、`Del` 删除、滚轮缩放、双击自适应 —— 这是标注工具的肌肉记忆。
3. **`shape_type` 的取值**（`polygon` / `rectangle` / `point` …），别自己发明一套。

**不抄的两样：**

1. **坐标约定**。X-AnyLabeling 的 `points` 是**绝对像素**（源码里没有任何 normalize）。
   我们的契约是**归一化**（`INTERFACE_ADDENDUM.md` §2.6）。
   互转就是 `× width / ÷ width`，但**必须封成一个转换函数**；混用一次就会出现
   "多边形整体偏移一个屏幕"这种最难查的 bug。
2. **整体架构**。它是 PyQt 桌面应用（`QGraphicsScene`），我们是 React + Konva。
   把 `label_widget.py`（上万行）搬过来只会把前端拖死。

**值得后续补**（按性价比排）：

- 导入/导出 X-AnyLabeling JSON（约半天，直接换来"与桌面工具互通"这个卖点）
- 手画多边形 + 拖顶点（P1 兜底路径；`/api/refine` 落地后优先级会降）
- 撤销栈目前存整份快照，实例上千时应改成 patch

---

## 已知取舍

- **打包 827 kB（gzip 266 kB）**，大头是 antd + konva。本机跑无感；
  要压的话把 antd 换成自写的 34 项搜索下拉，估计能省 ~400 kB。
- **`npm run build` 在受限 shell 里会 `spawn EPERM`**（Vite 8 的 `windowsSafeRealPathSync` 要 exec 子进程），
  用 `vite build --configLoader native` 可绕过。正常 PowerShell 里 `npm run dev` / `npm run build` 都没问题。
