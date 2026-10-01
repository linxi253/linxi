# Prepare a project-local official CPython runtime for native ONNX tooling.
# It reuses the project's pinned .venv packages and does not replace system DLLs.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runtimePath = Join-Path $projectRoot '.runtime/python310'
$pythonPath = Join-Path $runtimePath 'python.exe'
$expectedArchive = '608619f8619075629c9c69f361352a0da6ed7e62f83a0e19c63e0ea32eb7629d'
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot '.venv/Lib/site-packages'))) {
    throw 'First install the project training dependencies into .venv.'
}
if (-not (Test-Path -LiteralPath $pythonPath)) {
    New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null
    $archivePath = Join-Path $runtimePath 'python.zip'
    Invoke-WebRequest 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip' -OutFile $archivePath
    if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedArchive) {
        throw 'Official CPython archive digest changed; review before proceeding.'
    }
    Expand-Archive -LiteralPath $archivePath -DestinationPath $runtimePath -Force
}
@('python310.zip', '.', '../../.venv/Lib/site-packages', '../../src', 'import site') |
    Set-Content -LiteralPath (Join-Path $runtimePath 'python310._pth') -Encoding ascii
& $pythonPath -c "import sys; import atom_center; print(sys.version); print('atom-center', atom_center.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'CLI runtime check failed.' }
Write-Output "CLI runtime ready: $pythonPath"
