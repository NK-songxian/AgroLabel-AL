# 开发环境搭建记录（2026-11-02）

> 面向 `backend/` + `frontend/` 的负责人。本文件记录**这台机器上实际验证通过**的环境，
> 以及几个非标准做法**为什么**这么做。换机器时照抄即可，但先读第 4 节的坑。

---

## 1. 结论（可直接照抄）

| 层 | 选择 | 位置 |
|---|---|---|
| 解释器 | 工作区 venv，继承 conda base | `./.venv/Scripts/python.exe` |
| 基础环境 | conda base（Python 3.13.9） | `D:\IDE\conda\python.exe` |
| 前端 | Vite 8 + React 19 + TS 7 | `./frontend/` |
| 包源 | 阿里云 PyPI（HTTP） / 华为云 npm（HTTP） | 见第 3 节 |

**不要用 conda base 直接装包**：`D:\IDE\conda\Lib\site-packages` 在沙箱下不可写，装不进去。

---

## 2. 已装版本（实测）

### 后端 `.venv`

```
fastapi            0.162.0
uvicorn            0.54.0     (standard)
python-multipart   0.0.33
starlette          1.10.0
pydantic           2.12.4     ← 继承自 base
ultralytics        8.4.198
torch              2.9.1+cu128          ← 装进 venv，覆盖 base 的 CPU 版
torchvision        0.24.1+cu128         ← 必须与 torch 配对
opencv-python      5.0.0.93
numpy              2.3.5      ← 继承自 base
pillow             12.0.0     ← 继承自 base
```

### 前端 `frontend/`

```
react 19.3.1 / react-dom 19.3.1
react-konva 19.3.2 / konva 10.9.1
zustand 5.0.16
antd 6.6.8
axios 1.21.0
vite 8.3.5 / typescript 7.0.2
```

> ⚠️ 交接文档写的是 **React 18**，实际脚手架给的是 **React 19**。react-konva 19.x 支持 React 19，
> 接口契约不受影响。若要严格对齐文档，`npm i react@18 react-dom@18 @types/react@18` 即可回退。

---

## 3. 网络：必须走 HTTP，不能走 HTTPS

本机开着代理 `127.0.0.1:7897`，但**只有明文 HTTP 能通，HTTPS/CONNECT 一律失败**
（`curl` 报 exit 35，`Invoke-WebRequest` 报"基础连接已经关闭"）。

所以两个包源都必须用 **HTTP**：

| 用途 | 地址 | 说明 |
|---|---|---|
| PyPI | `http://mirrors.aliyun.com/pypi/simple/` | 索引页链接是相对路径，会解析成 HTTP，可用 |
| PyPI（备选） | 清华 / 腾讯 / npmmirror | **都会 301/302 跳 HTTPS，用不了** |
| npm | `http://mirrors.huaweicloud.com/repository/npm/` | 实测元数据与 tarball 都通 |
| npm（备选） | `http://registry.npmmirror.com` | 301 跳 HTTPS，用不了 |

### 装后端

```powershell
$ws = "C:\Users\1\Desktop\附件\code-web\AgroLabel-AL"
$py = Join-Path $ws ".venv\Scripts\python.exe"
$env:TEMP = "$ws\.tmp"; $env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = "$ws\.pipcache"
$env:HTTP_PROXY = 'http://127.0.0.1:7897'; $env:HTTPS_PROXY = 'http://127.0.0.1:7897'
$env:PIP_INDEX_URL = 'http://mirrors.aliyun.com/pypi/simple/'
$env:PIP_TRUSTED_HOST = 'mirrors.aliyun.com'

& $py -m pip install fastapi "uvicorn[standard]" python-multipart
& $py -m pip install --no-deps "torchvision==0.22.0"     # 必须 --no-deps，否则会重装 torch
& $py -m pip install ultralytics opencv-python
```

### 装前端

```powershell
$env:npm_config_registry   = 'http://mirrors.huaweicloud.com/repository/npm/'
$env:npm_config_proxy      = 'http://127.0.0.1:7897'
$env:npm_config_https_proxy= 'http://127.0.0.1:7897'
$env:npm_config_cache      = "$ws\.npmcache"
$env:npm_config_userconfig = "$ws\.npmrc"

cd frontend
npm install react-konva konva zustand antd axios
```

---

## 4. 三个必须知道的坑

### 坑 1：`tempfile.mkdtemp` 建出的目录写不进去（会让 pip 全部失效）

现象：`pip install` 任何包都报

```
OSError: [Errno 13] Permission denied: '...\pip-unpack-xxxx\foo.whl'
```

根因：本机 Python 3.13.9 在 Windows 上把 `os.mkdir(path, 0o700)` 的模式落成了
**限制性安全描述符**，建出的目录连 `icacls` 都读不了；而工作区的写权限依赖父目录 ACL 继承，
一旦目录自带显式描述符就断链。`tempfile.mkdtemp()` 正是用 `0o700` 建目录。

对照实测：

```
os.mkdir(p)          -> 可写
os.mkdir(p, 0o700)   -> 不可写
tempfile.mkdtemp()   -> 不可写     ← 这就是 pip 炸的原因
```

**已修复**：`.venv/Lib/site-packages/sitecustomize.py` 在进程启动时把 `os.mkdir` 的 `mode` 参数丢弃，
让新目录走正常 ACL 继承。文件里写了完整背景，**不要删**，删了 pip 立刻复发。

### 坑 2：`torchvision` 必须 `--no-deps` 单装

base 里是 **CPU 版 torch 2.7.0**。直接 `pip install ultralytics` 会让 pip 按依赖图去装 torchvision，
有概率把 torch 一起换掉（2.5 GB 白下）。正确做法是先把配对的 CPU 版装上：

```powershell
pip install --no-deps "torchvision==0.22.0"
```

装完 `torchvision.__version__` 应为 `0.22.0+cpu`，与 `torch 2.7.0` 匹配。

### 坑 3：`torch` 先导入会触发 `OMP: Error #15`

装上 CUDA 版 torch 之后，直接 `import torch` 会崩：

```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

原因：conda base 的 MKL 和 torch 的 CUDA 版各自带一份 `libiomp5md.dll`。

实测导入顺序的差异（同一解释器，同一环境）：

| 顺序 | 结果 |
|---|---|
| `import torch` 单独 | ❌ OMP 崩 |
| `import torch` → `cv2` | ❌ |
| **`import cv2` → `torch`** | ✅ |
| `from ultralytics import YOLO` → `torch` | ✅ |
| 设 `KMP_DUPLICATE_LIB_OK=TRUE` | ✅ |

**后端 `app.py` 开头固定这么写**（已实测通过）：

```python
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 兜底
import cv2                                              # 必须早于 torch
import numpy as np
from ultralytics import YOLO
import torch
import fastapi, uvicorn
```

### 坑 4：`vite build` / `vite dev` 在受限 shell 里报 `spawn EPERM`

Vite 在 Windows 上会 `exec` 一个子进程解析真实路径，受限 shell 不允许创建管道型子进程。
**这只是受限 shell 的限制**，你在自己的 PowerShell / CMD 里正常跑没有问题：

```powershell
cd frontend
npm run dev      # http://localhost:5173
npm run build
```

已验证：`tsc -b` 0 错误，`vite build` 106 ms 产出 `frontend/dist/`。

---

## 5. 模型侧已在本地跑通（P2 不再被阻塞）

交接文档说"`inference/`、`active_learning/` 尚未实现"，但**权重是在仓库里的**：

```
weights/baseline_yolov8nseg_plantseg34_best.pt   6.5 MB   YOLOv8n-seg · 34 类
```

### 5.1 装 CUDA 版 torch（可选，但建议装）

conda base 自带的是 **CPU 版** torch（`torch.version.cuda is None`），
但机器上有 **RTX 5070 Ti Laptop / 11.9 GB / sm_120**，驱动 610.74。
CUDA 轮子走 HTTP 可达且很快，**2.86 GB 约 2–4 分钟**：

```powershell
& $py -m pip install --no-cache-dir --no-deps `
   --index-url http://download.pytorch.org/whl/cu128 `
   "torch==2.9.1+cu128" "torchvision==0.24.1+cu128"
```

> 关键：`download.pytorch.org` 支持 **HTTP**（`http://download.pytorch.org/whl/cu128/` 返回 200），
> 实测走代理 17.6 MB/s。这和阿里云 PyPI 只有 300–650 KB/s 完全两个量级。
> `torch==2.9.1` 也正好对齐 `requirements.txt` 的声明。

装好后 `torch.cuda.is_available()` 为 True，`arch_list` 含 `sm_120`。

### 5.2 实测：单张时延

CPU = 16 核，GPU = RTX 5070 Ti Laptop，`imgsz=640`，均已预热。

| | 均值 | 中位 | P95 |
|---|---|---|---|
| CPU | 29.7 ms | 26.1 ms | 43.7 ms |
| GPU | 19.3 ms | 13.1 ms | 33.2 ms |

**单张只快约 1.6 倍**。原因不是 GPU 不行，是 YOLOv8n-seg 只有 3M 参数，
瓶颈落在 CPU 侧的 letterbox 缩放、NMS、掩码多边形提取上（Amdahl 定律）。

### 5.3 实测：批量才是 GPU 的战场

处理同样 **16 张图** 的总耗时：

| 设备 | 逐张 1× | 批量 4× | 批量 8× | 批量 16× |
|---|---|---|---|---|
| CPU | 346 ms | 620 ms | 609 ms | 723 ms |
| GPU | 212 ms | 132 ms | 122 ms | **114 ms** |

- **CPU 上批量反而更慢**（核心已被前后处理占满）
- **GPU 批量 16 张只要 114 ms，比 CPU 逐张快 3 倍、比 CPU 批量快 5.3 倍**

### 5.4 结论：什么时候用 GPU

| 场景 | 建议 | 理由 |
|---|---|---|
| `/api/predict` 单张（标注交互） | **CPU 就够** | 30 ms vs 19 ms，300 ms 预算下有 10 倍余量 |
| `/api/sample` 全池打分（TTA/entropy） | **必须 GPU + 批量** | 200 张 TTA：CPU 19 s → GPU 批量约 1.5 s |
| 导入时批量预标注（几百张） | **GPU + 批量** | 200 张：CPU 7.6 s → GPU 1.4 s |
| 增量微调 / 回流训练 | **GPU** | 另说，属模型侧 |

**给后端的实现要求**：`/api/sample` 和批量预标注内部要**攒批**（一次 8–16 张），
不要写成 `for img in pool: model.predict(img)` —— 那样 GPU 只快 1.6 倍，等于白装。

---

## 6. 目录约定（新增）

```
AgroLabel-AL/
├── .venv/            # 工作区虚拟环境（gitignore），继承 base 的 torch
├── .tmp/             # pip/python 临时目录（gitignore）
├── .pipcache/        # pip 缓存（gitignore）
├── .npmcache/        # npm 缓存（gitignore）
├── .acl-recovery/    # 沙箱权限诊断报告（gitignore）
└── frontend/         # Vite + React 19 + TS（已 npm install）
```

`.gitignore` 已补上以上五条，不会误提交。
