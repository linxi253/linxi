@echo off
chcp 65001 >nul
title STEM-HAADF 模拟工具 · stem_sim 引擎
cd /d "%~dp0"

rem 始终使用项目内 .venv，绝不回退到 PATH 上的 python。
rem 回退会带来两个后果：工具读到别的项目的依赖版本；缺失的包被自动装进
rem 全局解释器，从而污染本机所有其它工具。缺环境时直接报错并给出修复命令。
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

"%VENV_PY%" stem_tool\main.py
if errorlevel 1 pause
