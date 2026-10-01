param(
    [switch]$SkipSingleFile
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExecutable = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$sourceDirectory = Join-Path $repositoryRoot "src"
$entryScript = Join-Path $repositoryRoot "packaging\annotator_entry.py"
$versionFile = Join-Path $repositoryRoot "packaging\windows_version_info.txt"
$manifestFile = Join-Path $repositoryRoot "packaging\windows_app.manifest"
$guideFile = Join-Path $repositoryRoot "packaging\ANNOTATOR_GUIDE_zh-CN.txt"
$noticesFile = Join-Path $repositoryRoot "packaging\THIRD_PARTY_NOTICES.txt"
$assetDirectory = Join-Path $repositoryRoot "src\atom_center\assets"

if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw "未找到项目 Python 环境：$pythonExecutable"
}

$gitStatus = @(& git -C $repositoryRoot status --porcelain)
if ($LASTEXITCODE -ne 0) {
    throw "无法读取 Git 工作树状态"
}
if ($gitStatus.Count -gt 0) {
    throw "Git 工作树存在未提交改动。请先提交源码，再生成正式发布包。"
}

$version = (& $pythonExecutable -c "import atom_center; print(atom_center.__version__)").Trim()
if (-not $version) {
    throw "无法读取 atom_center 版本"
}

$buildRoot = Join-Path $repositoryRoot "build\annotator-$version"
$distRoot = Join-Path $repositoryRoot "dist\annotator-$version"
$releaseRoot = Join-Path $repositoryRoot "release"
$portableName = "AtomCenterAnnotator-v$version-win-x64"
$portableDirectory = Join-Path $releaseRoot $portableName
$portableZip = Join-Path $releaseRoot "$portableName.zip"
$singleExecutable = Join-Path $releaseRoot "AtomCenterAnnotator-v$version-win-x64-single.exe"
$hashFile = Join-Path $releaseRoot "AtomCenterAnnotator-v$version-SHA256SUMS.txt"

function Remove-VerifiedTree([string]$TargetPath) {
    $repositoryFull = [System.IO.Path]::GetFullPath($repositoryRoot).TrimEnd('\') + '\'
    $targetFull = [System.IO.Path]::GetFullPath($TargetPath)
    if (-not $targetFull.StartsWith($repositoryFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "拒绝删除仓库外路径：$targetFull"
    }
    if (Test-Path -LiteralPath $targetFull) {
        Remove-Item -LiteralPath $targetFull -Recurse -Force
    }
}

foreach ($path in @($buildRoot, $distRoot, $portableDirectory)) {
    Remove-VerifiedTree $path
}
foreach ($file in @($portableZip, $singleExecutable, $hashFile)) {
    if (Test-Path -LiteralPath $file) {
        Remove-Item -LiteralPath $file -Force
    }
}
New-Item -ItemType Directory -Force -Path $buildRoot, $distRoot, $releaseRoot | Out-Null
$specDirectory = Join-Path $buildRoot "specs"
New-Item -ItemType Directory -Force -Path $specDirectory | Out-Null

$commonArguments = @(
    "--noconfirm",
    "--clean",
    "--windowed",
    "--noupx",
    "--optimize", "1",
    "--specpath", $specDirectory,
    "--paths", $sourceDirectory,
    "--version-file", $versionFile,
    "--manifest", $manifestFile,
    "--hidden-import", "matplotlib.backends.backend_tkagg",
    "--hidden-import", "PIL._tkinter_finder",
    "--collect-data", "matplotlib",
    "--add-data", "$assetDirectory;atom_center/assets",
    "--exclude-module", "torch",
    "--exclude-module", "torchvision",
    "--exclude-module", "ultralytics",
    "--exclude-module", "onnx",
    "--exclude-module", "onnxruntime",
    "--exclude-module", "cv2",
    "--exclude-module", "scipy",
    "--exclude-module", "pytest"
)

Write-Host "[1/4] 构建便携文件夹版..."
$onedirArguments = $commonArguments + @(
    "--onedir",
    "--contents-directory", "_internal",
    "--name", "AtomCenterAnnotator",
    "--distpath", (Join-Path $distRoot "onedir"),
    "--workpath", (Join-Path $buildRoot "onedir"),
    $entryScript
)
& $pythonExecutable -m PyInstaller @onedirArguments
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller 便携版构建失败，退出码 $LASTEXITCODE"
}

$builtPortable = Join-Path $distRoot "onedir\AtomCenterAnnotator"
Copy-Item -LiteralPath $builtPortable -Destination $portableDirectory -Recurse
Copy-Item -LiteralPath $guideFile -Destination (Join-Path $portableDirectory "标注员使用说明.txt")
Copy-Item -LiteralPath $noticesFile -Destination (Join-Path $portableDirectory "THIRD_PARTY_NOTICES.txt")

$gitCommit = (& git -C $repositoryRoot rev-parse --short HEAD).Trim()
$pyInstallerVersion = (& $pythonExecutable -m PyInstaller --version).Trim()
$buildInfo = @(
    "Product: Atom Center Annotator",
    "Version: $version",
    "Platform: Windows x86-64",
    "Build time: $([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'))",
    "Git commit: $gitCommit",
    "Git tree state: clean",
    "Python: $(& $pythonExecutable -c 'import platform; print(platform.python_version())')",
    "PyInstaller: $pyInstallerVersion"
) -join "`r`n"
Set-Content -LiteralPath (Join-Path $portableDirectory "BUILD_INFO.txt") -Value $buildInfo -Encoding UTF8

Write-Host "[2/4] 验证便携文件夹版..."
$onedirReport = Join-Path $buildRoot "onedir-smoke.json"
$onedirProcess = Start-Process `
    -FilePath (Join-Path $portableDirectory "AtomCenterAnnotator.exe") `
    -ArgumentList @("--smoke-test", $onedirReport) `
    -Wait -PassThru -WindowStyle Hidden
if ($onedirProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $onedirReport)) {
    throw "便携版自检失败，退出码 $($onedirProcess.ExitCode)"
}
$onedirResult = Get-Content -LiteralPath $onedirReport -Raw | ConvertFrom-Json
if (-not $onedirResult.ok) {
    throw "便携版自检报告未通过：$onedirReport"
}
Copy-Item -LiteralPath $onedirReport -Destination (Join-Path $portableDirectory "SELF_TEST.json")

Compress-Archive -LiteralPath $portableDirectory -DestinationPath $portableZip -CompressionLevel Optimal

if (-not $SkipSingleFile) {
    Write-Host "[3/4] 构建并验证单文件版..."
    $onefileArguments = $commonArguments + @(
        "--onefile",
        "--name", "AtomCenterAnnotator_SingleFile",
        "--distpath", (Join-Path $distRoot "onefile"),
        "--workpath", (Join-Path $buildRoot "onefile"),
        $entryScript
    )
    & $pythonExecutable -m PyInstaller @onefileArguments
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller 单文件版构建失败，退出码 $LASTEXITCODE"
    }
    $builtSingle = Join-Path $distRoot "onefile\AtomCenterAnnotator_SingleFile.exe"
    Copy-Item -LiteralPath $builtSingle -Destination $singleExecutable
    $onefileReport = Join-Path $buildRoot "onefile-smoke.json"
    $onefileProcess = Start-Process `
        -FilePath $singleExecutable `
        -ArgumentList @("--smoke-test", $onefileReport) `
        -Wait -PassThru -WindowStyle Hidden
    if ($onefileProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $onefileReport)) {
        throw "单文件版自检失败，退出码 $($onefileProcess.ExitCode)"
    }
    $onefileResult = Get-Content -LiteralPath $onefileReport -Raw | ConvertFrom-Json
    if (-not $onefileResult.ok) {
        throw "单文件版自检报告未通过：$onefileReport"
    }
}

Write-Host "[4/4] 生成 SHA-256..."
$artifactPaths = @($portableZip)
if (-not $SkipSingleFile) {
    $artifactPaths += $singleExecutable
}
$hashLines = foreach ($artifact in $artifactPaths) {
    $hash = (Get-FileHash -LiteralPath $artifact -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $([System.IO.Path]::GetFileName($artifact))"
}
Set-Content -LiteralPath $hashFile -Value ($hashLines -join "`r`n") -Encoding ASCII

Write-Host "构建完成："
Write-Host "  便携版：$portableZip"
if (-not $SkipSingleFile) {
    Write-Host "  单文件：$singleExecutable"
}
Write-Host "  校验值：$hashFile"
