# All CLI arguments are forwarded literally to the project-local Python runtime.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonPath = Join-Path $projectRoot '.runtime/python310/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Run scripts/setup_cli_runtime.ps1 once before using this entry.'
}
& $pythonPath -X utf8 -m atom_center.cli @args
exit $LASTEXITCODE
