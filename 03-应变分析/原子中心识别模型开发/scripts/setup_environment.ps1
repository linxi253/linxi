$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    python -m venv (Join-Path $ProjectRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Creating .venv failed; Python 3.10 is required." }
}
& $VenvPython -m pip install "pip==25.1.1" "setuptools==84.0.0" "wheel==0.48.0"
if ($LASTEXITCODE -ne 0) { throw "Installing build tools failed." }
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements\training-win-py310.lock")
if ($LASTEXITCODE -ne 0) { throw "Installing the locked training dependencies failed." }
Push-Location $ProjectRoot
try {
    & $VenvPython -m pip install --no-deps --no-build-isolation -e .
    if ($LASTEXITCODE -ne 0) { throw "Installing atom-center failed." }
    & (Join-Path $PSScriptRoot "setup_cli_runtime.ps1")
    & (Join-Path $ProjectRoot ".runtime\python310\python.exe") -X utf8 (Join-Path $PSScriptRoot "check_environment.py") --require-gpu
    if ($LASTEXITCODE -ne 0) { throw "Training environment check failed." }
}
finally {
    Pop-Location
}
