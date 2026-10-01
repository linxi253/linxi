@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

REM ============================================================
REM  Version isolation: always build inside the bundled .venv-build,
REM  never with whatever "python" happens to be on PATH -- otherwise
REM  the same source produces a different exe on every machine.
REM  Dependencies are installed at the exact versions pinned in
REM  requirements-dev.lock.txt.
REM ============================================================
set "VENV=.venv-build"
set "BUILD_PYTHON=%VENV%\Scripts\python.exe"
set "BOOTSTRAP_PYTHON=python"
set "LOCK=requirements-dev.lock.txt"
REM The lock was resolved for this Python version (see the lock header).
set "WANT_PY=3.10"

echo ============================================
echo   Deloc Cleaner - Packaging Build
echo ============================================
echo.

if not exist "%LOCK%" (
    echo [x] Missing %LOCK% - cannot build reproducibly.
    echo     Regenerate it from requirements-dev.txt in a clean venv first.
    pause
    exit /b 1
)

echo [1/5] Preparing isolated build venv (%VENV%)...
if exist "%BUILD_PYTHON%" goto venv_ready
echo       creating venv with "%BOOTSTRAP_PYTHON%" ...
"%BOOTSTRAP_PYTHON%" -m venv "%VENV%"
if errorlevel 1 (
    echo       FAILED: could not create venv. Python %WANT_PY% must be on PATH.
    pause
    exit /b 1
)
:venv_ready
REM NOTE: deliberately no "for /f" here -- cmd mangles the nested quotes of
REM `for /f ... in ('"py.exe" -c "code"')` and the probe then fails outright.
"%BUILD_PYTHON%" -c "import platform;print('      build interpreter : Python', platform.python_version())"
if errorlevel 1 (
    echo       FAILED: cannot run %BUILD_PYTHON%
    pause
    exit /b 1
)
echo       build venv        : %VENV%
"%BUILD_PYTHON%" -c "import sys;sys.exit(0 if sys.version_info[:2]==(3,10) else 1)"
if not errorlevel 1 goto py_ok
echo       WARNING: lock was resolved for Python %WANT_PY%, building with another version.
:py_ok

echo.
echo [2/5] Installing locked dependencies (%LOCK%)...
REM -X utf8: pip reads requirements files with the locale encoding, which is
REM GBK on a Chinese Windows -- the lock header is UTF-8, so force UTF-8 mode.
"%BUILD_PYTHON%" -X utf8 -m pip install --disable-pip-version-check -q -r "%LOCK%"
if errorlevel 1 (
    echo       Dependency installation failed!
    pause
    exit /b 1
)

echo.
echo [3/5] Verifying the locked environment...
"%BUILD_PYTHON%" -m pip --version
"%BUILD_PYTHON%" -c "import numpy,scipy,tifffile,PIL,matplotlib,imagecodecs,PyInstaller;print('numpy',numpy.__version__,'| scipy',scipy.__version__,'| tifffile',tifffile.__version__,'| pillow',PIL.__version__,'| matplotlib',matplotlib.__version__,'| imagecodecs',imagecodecs.__version__,'| pyinstaller',PyInstaller.__version__)"
if errorlevel 1 (
    echo       FAILED: the locked dependency set is not importable.
    pause
    exit /b 1
)

echo.
echo [4/5] Running tests...
"%BUILD_PYTHON%" -X utf8 -m pytest tests -q
if errorlevel 1 (
    echo       Tests failed - build aborted.
    pause
    exit /b 1
)

echo.
echo [5/5] Building with PyInstaller...
"%BUILD_PYTHON%" -X utf8 -m PyInstaller --noconfirm --clean DelocCleaner.spec
if errorlevel 1 (
    echo       Build failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Build complete!
echo   Output: dist\DelocCleaner.exe
echo ============================================
pause
REM explicit success code: "pause" reports a non-zero code when stdin is
REM redirected (CI / automation), which would look like a failed build.
exit /b 0
