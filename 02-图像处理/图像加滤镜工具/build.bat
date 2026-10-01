@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

REM ============================================================
REM  Version isolation: always build inside the bundled .venv-build,
REM  never with whatever "python" happens to be on PATH -- otherwise
REM  the same source produces a different exe on every machine, and
REM  dependencies leak into the global interpreter.
REM  Build dependencies are pinned in requirements-dev.lock.txt.
REM ============================================================
set "BUILD_PYTHON=.venv-build\Scripts\python.exe"
set "BOOTSTRAP_PYTHON=python"
set "DEV_LOCK=requirements-dev.lock.txt"

echo ============================================
echo   TIF Image Filter Tool - Packaging Build
echo ============================================
echo.

if not exist "%DEV_LOCK%" (
    echo [x] Missing %DEV_LOCK% - cannot build reproducibly.
    echo     Regenerate it from requirements-dev.txt in a clean venv first.
    pause
    exit /b 1
)

echo [1/3] Preparing isolated build venv (.venv-build)...
if exist "%BUILD_PYTHON%" goto venv_ready
echo       creating venv with "%BOOTSTRAP_PYTHON%" ...
"%BOOTSTRAP_PYTHON%" -m venv .venv-build
if errorlevel 1 (
    echo       FAILED: could not create venv. Python 3.10 must be on PATH.
    pause
    exit /b 1
)
:venv_ready
"%BUILD_PYTHON%" -c "import sys;print('      build interpreter : Python',sys.version.split()[0])"

echo.
echo [2/3] Installing locked build dependencies (%DEV_LOCK%)...
REM -X utf8: pip reads requirements files with the locale encoding, which is
REM GBK on a Chinese Windows, while the lock header is UTF-8.
"%BUILD_PYTHON%" -X utf8 -m pip install --disable-pip-version-check -q -r "%DEV_LOCK%"
if errorlevel 1 (
    echo       Dependency installation failed!
    pause
    exit /b 1
)

echo.
echo [3/3] Running tests and building the versioned release...
"%BUILD_PYTHON%" -X utf8 build_release.py
if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Build complete!
echo   Output: dist\TIF_FilterTool.exe
echo ============================================
pause
