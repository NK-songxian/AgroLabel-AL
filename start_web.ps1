# AgroLabel-AL · 一键启动（Windows PowerShell）
#
# 用途：评审现场 / 离线机器上一条命令拉起整个系统。
# 前端已构建进 frontend/dist，由 FastAPI 一起托管 —— **不需要 Node，只用一个端口**。
#
# 用法：
#   .\start_web.ps1                      # 127.0.0.1:8000
#   .\start_web.ps1 -Port 8080           # 换端口
#   .\start_web.ps1 -HostAddr 0.0.0.0    # 让同网段其他机器也能访问（演示投屏用）
#   .\start_web.ps1 -NoBrowser           # 不自动开浏览器
#
# 依赖：Python 3.10+ 与 backend/requirements.txt。找不到 .venv 时退回系统 python。

param(
    [int]$Port = 8000,
    [string]$HostAddr = "127.0.0.1",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root

# ---------------------------------------------------------------- 找 Python
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $VenvPy) {
    $Py = $VenvPy
    Write-Host "[env] 用仓库内 .venv" -ForegroundColor DarkGray
} else {
    $Py = "python"
    Write-Host "[env] 未找到 .venv，退回系统 python（依赖需自行安装）" -ForegroundColor Yellow
}

# ---------------------------------------------------------------- 自检
& $Py -c "import fastapi, uvicorn, pydantic, PIL, cv2, yaml, ultralytics" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[×] 依赖不全。缺失时执行：" -ForegroundColor Red
    Write-Host "      $Py -m pip install -r backend\requirements.txt" -ForegroundColor Yellow
    Write-Host "    离线机器先建 wheelhouse：" -ForegroundColor DarkGray
    Write-Host "      pip download -r backend\requirements.txt -d wheelhouse" -ForegroundColor DarkGray
    Write-Host "      $Py -m pip install --no-index --find-links wheelhouse -r backend\requirements.txt" -ForegroundColor DarkGray
    Write-Host ""
    exit 1
}

if (-not (Test-Path -LiteralPath (Join-Path $Root "frontend\dist\index.html"))) {
    Write-Host "[!] frontend/dist 不存在，只能访问 /api/* 与 /docs" -ForegroundColor Yellow
    Write-Host "    联网机器上先构建：cd frontend; npm ci; npm run build" -ForegroundColor DarkGray
}

if (-not (Test-Path -LiteralPath (Join-Path $Root "weights"))) {
    Write-Host "[!] weights/ 不存在，/api/predict 会失败" -ForegroundColor Yellow
}

# ---------------------------------------------------------------- 环境变量
# 这三条 app.py 里也会 setdefault，这里显式设一遍，保证"启动前就生效"：
#   YOLO_OFFLINE=1        ultralytics 有联网自检（字体/AMP/权重探测），离线必须关掉
#   KMP_DUPLICATE_LIB_OK  conda MKL 与 CUDA 版 torch 各带一份 libiomp5md.dll
#   YOLO_CONFIG_DIR       ultralytics 想写用户配置目录，受限机器上会警告
$env:YOLO_OFFLINE = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
if (-not $env:YOLO_CONFIG_DIR) { $env:YOLO_CONFIG_DIR = Join-Path $Root ".tmp\ultralytics" }
New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null

# ---------------------------------------------------------------- 起服务
$Url = "http://{0}:{1}/" -f $(if ($HostAddr -eq "0.0.0.0") { "127.0.0.1" } else { $HostAddr }), $Port
Write-Host ""
Write-Host "  AgroLabel-AL" -ForegroundColor Cyan
Write-Host "  界面   $Url" -ForegroundColor Green
Write-Host "  接口   ${Url}api/health" -ForegroundColor DarkGray
Write-Host "  文档   ${Url}docs" -ForegroundColor DarkGray
Write-Host "  停止   Ctrl+C" -ForegroundColor DarkGray
Write-Host ""

if (-not $NoBrowser) {
    # 等服务起来再开浏览器（模型加载约 1~3 s）
    Start-Job -ScriptBlock {
        param($u)
        for ($i = 0; $i -lt 40; $i++) {
            try { Invoke-WebRequest "$u/api/health" -UseBasicParsing -TimeoutSec 2 | Out-Null; break }
            catch { Start-Sleep -Milliseconds 500 }
        }
        Start-Process $u
    } -ArgumentList $Url | Out-Null
}

& $Py -m uvicorn backend.app:app --host $HostAddr --port $Port
