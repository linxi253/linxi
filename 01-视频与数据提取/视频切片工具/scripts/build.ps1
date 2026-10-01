param(
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
# Version isolation: always build inside the project-local .venv, never with
# whatever "python" happens to be on PATH -- otherwise the same source produces
# a different exe on every machine, and missing dependencies leak into the
# global interpreter and break every other tool on the machine.
$venvPython = "$root\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw @"
Project virtual environment not found:
  $venvPython

Create it once with:
  python -m venv .venv
  .venv\Scripts\python -X utf8 -m pip install -r requirements.lock
"@
}
$python = $venvPython

# 版本单点来源：video_extractor/__init__.py 的 __version__（与 spec 保持一致）
$initSource = Get-Content -LiteralPath "$root\video_extractor\__init__.py" -Raw -Encoding UTF8
if ($initSource -notmatch '__version__ = "([^"]+)"') {
    throw "Cannot read __version__ from video_extractor\__init__.py"
}
$version = $Matches[1]
$releaseName = "TEMVideoExtractor-v$version"

$runtimeFiles = @(
    'ffmpeg.exe', 'ffprobe.exe',
    'avcodec-62.dll', 'avdevice-62.dll', 'avfilter-11.dll',
    'avformat-62.dll', 'avutil-60.dll', 'swresample-6.dll', 'swscale-9.dll'
)
$missingRuntimeFiles = @($runtimeFiles | Where-Object {
    -not (Test-Path -LiteralPath "$root\tools\ffmpeg\$_" -PathType Leaf)
})
if ($missingRuntimeFiles.Count -gt 0) {
    throw "Missing verified FFmpeg shared runtime files: $($missingRuntimeFiles -join ', '). Read tools\ffmpeg\README.md first."
}

# Supply-chain gate: verify every runtime file against the SHA-256 table in
# PROVENANCE.md. A swapped DLL must fail the build instead of shipping silently.
$provenancePath = "$root\tools\ffmpeg\PROVENANCE.md"
if (-not (Test-Path -LiteralPath $provenancePath -PathType Leaf)) {
    throw "Missing PROVENANCE.md; cannot verify FFmpeg runtime hashes."
}
$provenanceText = Get-Content -LiteralPath $provenancePath -Raw -Encoding UTF8
$expectedHashes = @{}
foreach ($match in [regex]::Matches($provenanceText, '\|\s*`([^`]+)`\s*\|\s*`([0-9a-fA-F]{64})`\s*\|')) {
    $expectedHashes[$match.Groups[1].Value] = $match.Groups[2].Value.ToLowerInvariant()
}
foreach ($name in $runtimeFiles) {
    if (-not $expectedHashes.ContainsKey($name)) {
        throw "PROVENANCE.md has no SHA-256 entry for $name; refusing to build."
    }
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath "$root\tools\ffmpeg\$name").Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHashes[$name]) {
        throw @"
FFmpeg runtime hash mismatch for $name
  expected: $($expectedHashes[$name])
  actual:   $actualHash
Replace the runtime from the verified source (see tools/ffmpeg/README.md).
"@
    }
}
Write-Host "FFmpeg runtime hashes verified against PROVENANCE.md."

& $python -m pip install -r requirements.lock
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed with exit code $LASTEXITCODE"
}
if (-not $SkipTests) {
    & $python -m pytest
    if ($LASTEXITCODE -ne 0) {
        throw "Test suite failed with exit code $LASTEXITCODE"
    }
}
& $python -m PyInstaller --noconfirm --clean .\tem_video_extractor_v4.spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}
$releaseExe = ".\dist\$releaseName.exe"
if (-not (Test-Path -LiteralPath $releaseExe -PathType Leaf)) {
    throw "Release executable was not created: $releaseExe"
}
$releaseHash = (Get-FileHash -Algorithm SHA256 $releaseExe).Hash.ToLowerInvariant()
"$releaseHash  $releaseName.exe" |
    Set-Content -Encoding ascii ".\dist\$releaseName.sha256.txt"

$smokeProcess = Start-Process `
    -FilePath $releaseExe `
    -ArgumentList '--headless-smoke' `
    -WindowStyle Hidden `
    -Wait `
    -PassThru
if ($smokeProcess.ExitCode -ne 0) {
    throw "Single-file release smoke test failed with exit code $($smokeProcess.ExitCode)"
}
