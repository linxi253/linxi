@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 始终使用项目内 .venv，绝不回退到 PATH 上的 python：
rem 回退会让工具读到别的项目的依赖版本，并把缺失的包装进全局解释器。
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [错误] 未找到项目虚拟环境：
    echo        %VENV_PY%
    echo.
    echo 首次使用请先创建：
    echo     python -m venv .venv
    echo     .venv\Scripts\python -X utf8 -m pip install -r requirements.lock.txt
    pause
    exit /b 1
)

"%VENV_PY%" atomic_app.py
if errorlevel 1 pause
