@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ═══════════════════════════════════════════════
echo   电镜晶体/非晶区域统计分析工具 v5.3
echo   依赖安装脚本（安装到项目独立虚拟环境 .venv）
echo ═══════════════════════════════════════════════
echo.

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple"

echo [1/3] 准备项目独立虚拟环境 (.venv)...
if exist "%VENV_PY%" goto :venv_ready
python -m venv .venv
if errorlevel 1 (
    echo.
    echo ✖ 创建虚拟环境失败！请确认 PATH 上有 Python 3.10。
    pause
    exit /b 1
)
:venv_ready
"%VENV_PY%" -c "import sys;print('      解释器: Python',sys.version.split()[0])"

echo.
echo [2/3] 安装锁定依赖（requirements.lock.txt）...
"%VENV_PY%" -X utf8 -m pip install --disable-pip-version-check -r requirements.lock.txt -i %MIRROR%
if errorlevel 1 (
    echo.
    echo ✖ 安装失败！请检查网络连接，或手动执行:
    echo   .venv\Scripts\python -m pip install -r requirements.lock.txt
    pause
    exit /b 1
)

echo.
echo [3/3] 验证安装...
"%VENV_PY%" -c "import cv2, numpy, scipy, tifffile, pandas, openpyxl, ttkbootstrap; print('✔ 所有依赖安装成功！')"
if errorlevel 1 (
    echo ✖ 部分依赖验证失败，请检查上方错误信息
    pause
    exit /b 1
)

echo.
echo ✔ 安装完成！现在可以运行:
echo   .venv\Scripts\python main.py
echo.
echo 注意：请始终使用上面的 .venv 解释器，不要直接用 PATH 上的 python，
echo       否则会读到别的项目的依赖版本。
echo.
pause
