# 本地启动说明（临时方案）

> **当前阶段：本机临时使用。**
> 依赖封装（wheelhouse / 离线安装包 / 免安装便携版）**等所有功能做完再统一做**，
> 见第 8 节。这份说明只解决一件事：**在仓库里一条命令把系统跑起来**。
>
> 长期方案（正式交付用）留档在同目录 `OFFLINE_DEPLOY_FULL.md`（依赖清单与 wheelhouse 做法）。
> 写于 2026-11-04，与当时代码实测一致。

---

## 0. 一句话

前端已构建进 `frontend/dist`，由 **FastAPI 一起托管**：
**一个进程、一个端口**，浏览器开 `http://127.0.0.1:8000/` 即可。

```
浏览器 ──HTTP──> FastAPI :8000 ──进程内──> ultralytics YOLO-seg
                    ├─ /            frontend/dist/index.html
                    ├─ /assets/*    打包好的 js / css
                    ├─ /api/*       11 个接口
                    └─ /docs        Swagger
```

现在的用法**不需要 Node**，也不再需要另起 4173 预览服务。

---

## 1. 相对地址约定（为什么不用绝对地址）

三条都刻意做成"跟着仓库走"，换机器、换目录、换端口都不用改配置：

| 位置 | 做法 | 好处 |
|---|---|---|
| 所有命令 | 在**仓库根目录**执行，用 `.\start_web.ps1`、`python -m uvicorn ...` 这类相对写法 | 文档里不出现 `C:\Users\...`，别人拷走就能用 |
| 启动脚本 | `$Root = $PSScriptRoot`，不依赖当前工作目录 | 从任何地方 `& "D:\xxx\start_web.ps1"` 也能跑 |
| 前端 API | `const BASE = import.meta.env.VITE_API_BASE ?? ''` → fetch 打的是 `/api/...` **相对路径** | 同源部署下自动跟着当前 host/port；也能直接挂到任意反代后面 |

> 例外：`vite dev`（5173）需要把 `/api` 代理到 8000，已在 `vite.config.ts` 里配好，
> 目标可用 `VITE_API_TARGET` 覆盖。生产构建（`npm run build`）不受影响，仍是相对路径。

**所以投屏演示可以直接换地址**：

```powershell
.\start_web.ps1 -HostAddr 0.0.0.0 -Port 8080
# 别的机器开 http://<你的内网IP>:8080/  —— 前端会自动打 /api，不需要改任何配置
```

---

## 2. 启动

### 双击（推荐，不用输命令）

打开仓库根目录，**双击 `启动.bat`**。

会弹出一个黑窗口，自动做完：找 `.venv` → 自检依赖 → 设环境变量 →
起服务 → 打开浏览器。窗口里会打印

```
  AgroLabel-AL
  界面   http://127.0.0.1:8000/
  接口   http://127.0.0.1:8000/api/health
  文档   http://127.0.0.1:8000/docs
  停止   Ctrl+C
```

**关窗口或按 Ctrl+C 就是停止服务。**
启动失败时窗口不会自动关闭（`pause`），把错误截图发群里即可。

带参数也能用（例如换端口）：

```bat
启动.bat -Port 8080
启动.bat -HostAddr 0.0.0.0
```

### 命令行（等价）

```powershell
# 仓库根目录
.\start_web.ps1
```

> 这台机器只有 Windows PowerShell 5.1（没有 `pwsh`），上面的写法在 5.1 下可用。
> `启动.bat` 已带 `-ExecutionPolicy Bypass`，双击不受执行策略限制。
> 若自己写命令被拦住：
> ```powershell
> powershell -NoProfile -ExecutionPolicy Bypass -File .\start_web.ps1
> ```
> 从任意目录启动也可以（脚本内部用 `$PSScriptRoot` 定位仓库）：
> ```powershell
> & "D:\某处\AgroLabel-AL\start_web.ps1" -Port 8099
> ```

脚本做四件事：找 `.venv` → 自检依赖 → 设环境变量 → 起 uvicorn 并开浏览器。

```powershell
.\start_web.ps1 -Port 8080            # 换端口
.\start_web.ps1 -HostAddr 0.0.0.0     # 同网段可访问（投屏用）
.\start_web.ps1 -NoBrowser            # 不自动开浏览器
```

手动起也行（脚本只是包了一层）：

```powershell
$env:YOLO_OFFLINE="1"; $env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

> **本阶段直接用仓库内 `.venv`**（它继承 conda base 的 torch，只补了服务端依赖）。
> 换机器时如果 `.venv` 不在，脚本会退回系统 `python` 并提示缺哪些包。

---

## 3. 环境变量

| 变量 | 默认 | 何时要改 |
|---|---|---|
| `AGROLABEL_WEIGHTS` | `weights/baseline_yolov8nseg_plantseg34_best.pt` | 换最优权重（模型侧出分后重测 §08 要用） |
| `AGROLABEL_DATA_ROOT` | 自动探测 | 数据放别处时指过去；有 `images/`+`data.yaml` 按 YOLO 处理，否则按 COCO 清单 |
| `AGROLABEL_ARM` | `assisted` | 三臂实验的默认臂（前端也会传，一般不用改） |
| `AGROLABEL_DEVICE` | `0` | 无 GPU 时设 `cpu` |
| `AGROLABEL_CONF` / `AGROLABEL_IMGSZ` | `0.25` / `640` | 一般不动 |
| `YOLO_OFFLINE` | `1`（`app.py` 里 setdefault） | **必须为 1**：ultralytics 有联网自检（字体/AMP/权重探测），关掉才不会卡 |
| `KMP_DUPLICATE_LIB_OK` | `TRUE`（`app.py` 里 setdefault） | Windows + conda MKL + CUDA torch 冲突时必须 |
| `YOLO_CONFIG_DIR` | `start_web.ps1` 设成 `.tmp/ultralytics` | 受限目录下 ultralytics 写用户配置失败时 |

这些变量**只影响本机当前会话**，脚本里设完不会污染系统环境。

---

## 4. 数据源三种模式

启动时按顺序探测，`/api/health` 的 `dataset.mode` 会告诉你走的是哪条：

| 顺序 | 条件 | mode | 说明 |
|---|---|---|---|
| 1 | `AGROLABEL_DATA_ROOT` 指向的目录 | `yolo` / `manifest` | 有 `images/`+`data.yaml` 判为 yolo |
| 2 | `configs/dataset_plantseg34.local.yaml` 里的 `path` | 同上 | |
| 3 | `datasets/plantseg100_pack/` | `yolo` | **当前本地演示走这条**（100 张 × 34 类） |
| 4 | `out/demo/` 或 `experiments/sam3_sample_vis/` | `demo` | 只有样例图，**不能用来跑实验** |

> `datasets/` 整包在 `.gitignore` 里，**不会随仓库分发**。
> 换机器要单独拷 `datasets/plantseg100_pack/`，否则会退到 demo 模式。

---

## 5. 启动后自检（30 秒）

```powershell
# 1) 服务与数据源
Invoke-RestMethod http://127.0.0.1:8000/api/health |
  Select-Object status, classes, @{n='mode';e={$_.dataset.mode}}, @{n='total';e={$_.dataset.total}}

# 期望：status=ok  classes=34  mode=yolo  total=100
```

```powershell
# 2) 界面能开
Invoke-WebRequest http://127.0.0.1:8000/ -UseBasicParsing | Select-Object StatusCode, @{n='type';e={$_.Headers['Content-Type']}}

# 期望：200 + text/html
```

```powershell
# 3) 真模型能跑（确认权重加载成功）
$body = @{ image_id = "apple_black_rot_1" } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/predict -Method Post -Body $body -ContentType "application/json" |
  Select-Object latency_ms, @{n='n';e={$_.instances.Count}}

# 期望：latency_ms 在 50~150 之间，n >= 1
```

```powershell
# 4) 完整冒烟（128 项断言，会写临时文件但自动还原）
python backend/tests/smoke_test.py
```

---

## 6. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `/` 返回 JSON 而不是界面 | `frontend/dist` 缺失 | `cd frontend; npm ci; npm run build` |
| 界面能开但图裂 / 请求全 404 | 前端从**别的端口**打开（如 4173）却打相对 `/api` | 本阶段统一从 `:8000` 打开；或 `npm run dev` 走 vite 代理 |
| `/api/predict` 报 500 且提到权重 | `weights/*.pt` 没带 | 拷权重；或设 `AGROLABEL_WEIGHTS` |
| 启动卡住十几秒无输出 | `YOLO_OFFLINE` 没生效，ultralytics 在联网探测 | 用 `start_web.ps1`，或手动 `$env:YOLO_OFFLINE="1"` |
| `OMP: Error #15` | conda MKL 与 CUDA torch 各带一份 `libiomp5md.dll` | 保持 `KMP_DUPLICATE_LIB_OK=TRUE`（`app.py` 已设） |
| 脚本报语法错 / 中文乱码 | PowerShell 5.1 把无 BOM 的 UTF-8 当 ANSI 读 | `start_web.ps1` 已存为 **UTF-8 with BOM**，改它时别丢 BOM |
| 端口被占 | 上次进程没退 | 换 `启动.bat -Port 8081`，或 `Get-NetTCPConnection -LocalPort 8000` 找进程 |
| 双击黑窗一闪就没 | 启动脚本报错退出 | 窗口本该 `pause`；若仍闪退，用命令行跑 `启动.bat` 看报错 |

---

## 7. 与开发环境的区别

`docs/ENV_SETUP.md` 记的是**开发机怎么联网装环境**（含代理和四个坑）。两者不重复：

| | 开发（ENV_SETUP.md） | 本说明（临时本地运行） |
|---|---|---|
| 网络 | 需要（走 HTTP 代理） | **不需要** |
| Node | 需要（vite dev / build） | **不需要**（除非要重新构建） |
| 端口 | 5173(dev) + 8000(api) | **只要 8000** |
| 前端 | `npm run dev` 热更新 | `dist` 由 FastAPI 静态托管 |

---

## 8. 后续（正式交付时再做）

**依赖封装暂缓**，等所有功能定稿后统一处理，内容包括：

- [ ] 建 `wheelhouse/`（`pip download -r backend/requirements.txt -d wheelhouse`，
      含 ultralytics 的传递依赖；torch 按目标机是否已有 CUDA 版决定）
- [ ] 目标机离线安装：`pip install --no-index --find-links wheelhouse -r backend/requirements.txt`
- [ ] 可选：便携式 Python 运行时（免安装 conda/miniconda 打包）
- [ ] 可选：把 `weights/` + `datasets/plantseg100_pack/` + `frontend/dist/` 打成一个发布 zip
- [ ] 目标机验收：按第 5 节四条自检全绿

> 现在不做的原因：功能仍在改（三臂实验、`human_stats.py` 等未完成），
> 提前锁 wheelhouse 等于每改一次依赖就重打一次包，浪费时间。

依赖清单与 wheelhouse 的完整做法见 **`docs/OFFLINE_DEPLOY_FULL.md`**。
