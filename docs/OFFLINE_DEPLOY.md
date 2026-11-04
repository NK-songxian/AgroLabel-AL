# 离线部署说明（AgroLabel-AL）

> 面向：评审现场 / 内网机器 / 断网环境演示。
> 目标：**一条命令拉起系统，只用一个端口**。
> 写于 2026-11-04，与当时代码实测一致。

---

## 0. 一句话

前端已经构建成静态文件（`frontend/dist`），由 **FastAPI 一起托管**：
目标机器只需要 **Python**，**不需要 Node**，浏览器开 `http://127.0.0.1:8000/` 即可。

```
浏览器 ──HTTP──> FastAPI :8000 ──进程内──> ultralytics YOLO-seg
                    ├─ /            frontend/dist/index.html
                    ├─ /assets/*    打包好的 js / css
                    ├─ /api/*       11 个接口
                    └─ /docs        Swagger
```

---

## 1. 要带什么（离线包清单）

| 内容 | 路径 | 体积 | 必需 | 说明 |
|---|---|---|---|---|
| 代码 | 整个仓库 | ~30 MB | ✅ | 排除 `.git`、`.venv`、`node_modules` |
| 基线权重 | `weights/baseline_yolov8nseg_plantseg34_best.pt` | **6.5 MB** | ✅ | 已随仓库分发 |
| 前端构建产物 | `frontend/dist/` | **0.8 MB** | ✅ | `index.html` + `assets/*.js/css` |
| 数据集采样包 | `datasets/plantseg100_pack/` | **14.6 MB** | ✅ | 100 张 × 34 类，含 `al_pool/` |
| Python wheel 包 | `wheelhouse/`（自建） | ~1.5 GB | ✅ | 见第 2 节，含 torch |
| Node 运行时 | — | — | ❌ | **不需要**，除非现场要重新构建前端 |

> 前端什么时候要重建：改了 `frontend/src` 之后。纯演示/部署不需要 Node。

---

## 2. 装依赖（离线）

### 2.1 在**联网**机器上建 wheelhouse

```bash
# 与目标机器同 OS、同 CPU 架构、同 Python 大版本（本项目实测 Python 3.13 / Windows AMD64）
pip download -r backend/requirements.txt -d wheelhouse

# torch 不在 requirements 里（通常由目标机的 conda 提供）。若目标机没有：
pip download torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128 -d wheelhouse
```

`pip download` 会**连传递依赖一起**下（matplotlib/pandas/scipy/psutil/polars 等），
所以 wheelhouse 体积大是正常的。

### 2.2 在**离线**机器上装

```bash
python -m pip install --no-index --find-links wheelhouse -r backend/requirements.txt
```

验证：

```bash
python -c "import fastapi, uvicorn, pydantic, PIL, cv2, yaml, ultralytics; print('deps ok')"
```

### 2.3 用现成 venv 的捷径

仓库里的 `.venv` 是"继承 conda base 的 torch，只补服务端依赖"的做法。
如果目标机器的 conda 环境已有 CUDA 版 torch，**只需要补这几条**：

```bash
pip install --no-index --find-links wheelhouse \
  fastapi==0.162.0 "uvicorn[standard]==0.54.0" pydantic==2.12.4 \
  Pillow==12.0.0 opencv-python==5.0.0 numpy==2.3.5 PyYAML==6.0.3 ultralytics==8.4.198
```

---

## 3. 启动

```powershell
.\start_web.ps1
```

脚本会做四件事：找 `.venv` → 自检依赖 → 设环境变量 → 起 uvicorn 并开浏览器。

常用参数：

```powershell
.\start_web.ps1 -Port 8080            # 换端口
.\start_web.ps1 -HostAddr 0.0.0.0     # 同网段可访问（投屏演示）
.\start_web.ps1 -NoBrowser            # 不自动开浏览器
```

手动起也可以（脚本只是包了一层）：

```powershell
$env:YOLO_OFFLINE="1"; $env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

---

## 4. 环境变量

| 变量 | 默认 | 何时要改 |
|---|---|---|
| `AGROLABEL_WEIGHTS` | `weights/baseline_yolov8nseg_plantseg34_best.pt` | 换最优权重（模型侧出分后重测 §08 要用） |
| `AGROLABEL_DATA_ROOT` | 自动探测 | 数据放在别处时指过去；有 `images/`+`data.yaml` 按 YOLO 处理，否则按 COCO 清单 |
| `AGROLABEL_ARM` | `assisted` | 三臂实验的默认臂（前端也会传，一般不用改） |
| `AGROLABEL_DEVICE` | `0` | 无 GPU 时设 `cpu` |
| `AGROLABEL_CONF` / `AGROLABEL_IMGSZ` | `0.25` / `640` | 一般不动 |
| `YOLO_OFFLINE` | `1`（`app.py` 里 setdefault） | **离线必须为 1** |
| `KMP_DUPLICATE_LIB_OK` | `TRUE`（`app.py` 里 setdefault） | Windows + conda MKL + CUDA torch 冲突时必须 |
| `YOLO_CONFIG_DIR` | `start_web.ps1` 设成 `.tmp/ultralytics` | 受限机器上 ultralytics 写用户目录失败时 |

`YOLO_OFFLINE=1` 为什么关键：ultralytics 启动时有一些**联网自检**（字体、AMP 校验、
权重探测）。断网环境下它会卡住或抛异常，而不是干净地报错。

---

## 5. 数据源三种模式

后端启动时按顺序探测，`/api/health` 的 `dataset.mode` 会告诉你走的是哪条：

| 顺序 | 条件 | mode | 说明 |
|---|---|---|---|
| 1 | `AGROLABEL_DATA_ROOT` 指向的目录 | `yolo` / `manifest` | 有 `images/`+`data.yaml` 判为 yolo |
| 2 | `configs/dataset_plantseg34.local.yaml` 里的 `path` | 同上 | |
| 3 | `datasets/plantseg100_pack/` | `yolo` | **当前离线演示走这条** |
| 4 | `out/demo/` 或 `experiments/sam3_sample_vis/` | `demo` | 只有样例图，**不能用来跑实验** |

> ⚠️ `datasets/` 整包在 `.gitignore` 里，**不会随仓库分发**。
> 离线包要单独拷 `datasets/plantseg100_pack/` 过去，否则会退到 demo 模式。

---

## 6. 启动后自检（30 秒）

```powershell
# 1) 服务与数据源
Invoke-RestMethod http://127.0.0.1:8000/api/health |
  Select-Object status, classes, @{n='mode';e={$_.dataset.mode}}, @{n='total';e={$_.dataset.total}}

# 期望：status=ok  classes=34  mode=yolo  total=100
```

```powershell
# 2) 界面能开
Invoke-WebRequest http://127.0.0.1:8000/ -UseBasicParsing | Select-Object StatusCode

# 3) 真模型能跑（关键：确认权重加载成功）
$body = @{ image_id = "apple_black_rot_1" } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/predict -Method Post -Body $body `
  -ContentType "application/json" |
  Select-Object latency_ms, @{n='n';e={$_.instances.Count}}

# 期望：latency_ms 在 50~150 之间，n >= 1
```

```powershell
# 4) 完整冒烟（128 项断言，会写临时文件但自动还原）
python backend/tests/smoke_test.py
```

第 4 步会动 `out/`（写入后还原），**离线演示前建议跑一次**——
它覆盖了 11 个接口、三臂目录隔离、导出、点提示、编辑代价。

---

## 7. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `/` 返回 JSON 而不是界面 | `frontend/dist` 缺失 | 联网机器 `npm ci && npm run build`，把 `dist/` 拷过来 |
| `/api/predict` 报 500 且提到权重 | `weights/*.pt` 没带 | 拷权重；或设 `AGROLABEL_WEIGHTS` |
| 启动卡住十几秒无输出 | `YOLO_OFFLINE` 没生效，ultralytics 在联网探测 | 用 `start_web.ps1` 启动，或手动 `set YOLO_OFFLINE=1` |
| `OMP: Error #15` | torch 先于 cv2 导入 / MKL 冲突 | 保持 `KMP_DUPLICATE_LIB_OK=TRUE`（`app.py` 已设） |
| `data.yaml 里 train 没有图片` | `m6_editcost` 的相对路径问题 | 已修（相对路径按 yaml 所在目录解析）；更新代码即可 |
| 界面能开但图裂 | `/api/images/{id}/file` 失败 | 检查数据集路径与 `mode` |
| 端口被占 | 上次进程没退 | 换 `-Port 8081`，或 `Get-NetTCPConnection -LocalPort 8000` 找进程 |

---

## 8. 与开发环境的区别

`docs/ENV_SETUP.md` 记的是**开发机怎么联网装环境**（含代理和四个坑）。
本文是**目标机器怎么断网跑起来**。两者不重复：

| | 开发（ENV_SETUP.md） | 部署（本文） |
|---|---|---|
| 网络 | 需要（走 HTTP 代理 127.0.0.1:7897） | **不需要** |
| Node | 需要（vite dev / build） | **不需要** |
| 端口 | 5173(dev) + 8000(api) | **只要 8000** |
| 前端 | `npm run dev` 热更新 | `dist` 静态托管 |
