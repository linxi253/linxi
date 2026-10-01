param(
    [string]$CertificateThumbprint = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$ProjectRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "..")
)
$BuildVenv = Join-Path $ProjectRoot ".venv-build"
$Python = Join-Path $BuildVenv "Scripts\python.exe"
$ReleaseDir = Join-Path $ProjectRoot "release"
$WorkDir = Join-Path $ProjectRoot "build\v2.1"

function Assert-NativeSuccess {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    python -m venv $BuildVenv
    Assert-NativeSuccess "Create build virtual environment"
}

& $Python -c "import struct, sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) and struct.calcsize('P') == 8 else 1)"
Assert-NativeSuccess "Require 64-bit CPython 3.10 for the locked Windows build"

& $Python -m pip install --upgrade "pip==26.1.2" "setuptools==83.0.0"
Assert-NativeSuccess "Install patched build bootstrap"
& $Python -m pip install --require-hashes `
    -r (Join-Path $ProjectRoot "requirements-win-py310.lock")
Assert-NativeSuccess "Install locked runtime dependencies"
& $Python -m pip install `
    "pyinstaller==6.21.0" "pyinstaller-hooks-contrib==2026.6" `
    "altgraph==0.17.5" "packaging==26.2" "pefile==2024.8.26" `
    "pywin32-ctypes==0.2.3" "pip-audit==2.10.1" "ruff==0.16.0"
Assert-NativeSuccess "Install pinned release tools"

Push-Location $ProjectRoot
try {
    & $Python -m pip check
    Assert-NativeSuccess "Dependency consistency check"
    & $Python -m pip_audit --requirement `
        ".\requirements-win-py310.lock" --disable-pip
    Assert-NativeSuccess "Dependency vulnerability audit"
    & $Python -m pip_audit --local --progress-spinner off
    Assert-NativeSuccess "Build-environment vulnerability audit"
    & $Python -m ruff check .
    Assert-NativeSuccess "Static analysis"
    & $Python -m unittest discover -s tests -v
    Assert-NativeSuccess "Regression tests"
    & $Python -m PyInstaller --clean --noconfirm `
        --distpath $ReleaseDir `
        --workpath $WorkDir `
        ".\stem-optimize-v2.1.spec"
    Assert-NativeSuccess "PyInstaller build"

    $Executables = @(
        Get-ChildItem -LiteralPath $ReleaseDir -Filter "*.exe" -File
    )
    if ($Executables.Count -ne 1) {
        throw "Expected exactly one release executable; found $($Executables.Count)."
    }
    $Executable = $Executables[0].FullName

    $SelfTest = Start-Process -FilePath $Executable `
        -ArgumentList "--self-test" -Wait -PassThru -WindowStyle Hidden
    if ($SelfTest.ExitCode -ne 0) {
        throw "Packaged runtime self-test failed with exit code $($SelfTest.ExitCode)."
    }

    if ($CertificateThumbprint) {
        $SignTool = Get-Command "signtool.exe" -ErrorAction Stop
        & $SignTool.Source sign /sha1 $CertificateThumbprint /fd SHA256 `
            /tr "http://timestamp.digicert.com" /td SHA256 $Executable
        Assert-NativeSuccess "Code signing"
        $Signature = Get-AuthenticodeSignature -LiteralPath $Executable
        if ($Signature.Status -ne "Valid") {
            throw "Code signature validation failed: $($Signature.Status)"
        }
    }

    $Hash = Get-FileHash -LiteralPath $Executable -Algorithm SHA256
    $HashLine = "$($Hash.Hash.ToLowerInvariant())  $($Hash.Path | Split-Path -Leaf)"
    [System.IO.File]::WriteAllText(
        (Join-Path $ReleaseDir "SHA256SUMS.txt"),
        "$HashLine`n",
        [System.Text.UTF8Encoding]::new($false)
    )

    $Signature = Get-AuthenticodeSignature -LiteralPath $Executable
    Write-Host "Release: $Executable"
    Write-Host "SHA256: $($Hash.Hash)"
    Write-Host "Signature: $($Signature.Status)"
    if (-not $CertificateThumbprint) {
        Write-Warning "No code-signing certificate was supplied; artifact is unsigned."
    }
}
finally {
    Pop-Location
}
