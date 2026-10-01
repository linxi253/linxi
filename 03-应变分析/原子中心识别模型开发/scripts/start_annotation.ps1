param(
    [ValidateSet("haadf_stem", "hrtem")]
    [string]$Modality = "haadf_stem"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExecutable = Join-Path $repositoryRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw "未找到项目 Python 环境。请先运行 scripts\setup_environment.ps1"
}

$imageDirectory = Join-Path $repositoryRoot "data\raw\$Modality"
$labelDirectory = Join-Path $repositoryRoot "data\labels\$Modality"
$projectDirectory = Join-Path $repositoryRoot "data\annotation_projects"
$manifestDirectory = Join-Path $repositoryRoot "data\manifests"
$projectFile = Join-Path $projectDirectory "$Modality.json"
$manifestFile = Join-Path $manifestDirectory "${Modality}_annotations.csv"

foreach ($directory in @($imageDirectory, $labelDirectory, $projectDirectory, $manifestDirectory)) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}

Write-Host "原始图像目录: $imageDirectory"
Write-Host "推荐目录层级: <sample_id>\<acquisition_id>\image.tif"

& $pythonExecutable -m atom_center.annotator_gui `
    --project $projectFile `
    --images $imageDirectory `
    --labels $labelDirectory `
    --manifest $manifestFile `
    --modality $Modality

exit $LASTEXITCODE
