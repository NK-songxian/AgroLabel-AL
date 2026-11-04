@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

title AgroLabel-AL

rem ============================================================
rem  双击即可启动。真正的逻辑在 start_web.ps1，这里只做三件事：
rem    1) 切到本文件所在目录（避免从别处运行时找不到 backend/）
rem    2) 设两个必需的环境变量
rem    3) 调 PowerShell 跑 start_web.ps1
rem  参数会透传，例如： 启动.bat -Port 8080 -NoBrowser
rem ============================================================

set "YOLO_OFFLINE=1"
set "KMP_DUPLICATE_LIB_OK=TRUE"

if not exist "%~dp0start_web.ps1" (
    echo.
    echo [x] 找不到 start_web.ps1，请确认本文件在仓库根目录。
    echo.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_web.ps1" %*

set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" (
    echo.
    echo [x] 启动失败，退出码 %CODE%。上面的错误信息可截图发群里。
    echo.
    pause
)
endlocal
