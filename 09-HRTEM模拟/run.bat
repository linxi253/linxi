@echo off
rem HRTEM simulation tool launcher (ASCII only to avoid GBK mojibake)
rem
rem Runs inside the project-local .venv. It must never fall back to whatever
rem "python" happens to be on PATH, nor to the Miniconda base interpreter:
rem doing so lets this tool read another project's dependency versions and
rem install missing packages into the global interpreter, which then breaks
rem every other tool on the machine.
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [ERROR] Project virtual environment not found:
    echo         %VENV_PY%
    echo.
    echo Create it once with:
    echo     python -m venv .venv
    echo     .venv\Scripts\python -X utf8 -m pip install -r requirements.lock.txt
    pause
    exit /b 1
)

"%VENV_PY%" hrtem_tool\main.py
if errorlevel 1 pause
