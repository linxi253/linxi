$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExecutable = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$releaseRoot = Join-Path $repositoryRoot "release"
if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw "未找到项目 Python 环境：$pythonExecutable"
}
$version = (& $pythonExecutable -c "import atom_center; print(atom_center.__version__)").Trim()
$zipPath = Join-Path $releaseRoot "AtomCenterAnnotator-v$version-win-x64.zip"
$singlePath = Join-Path $releaseRoot "AtomCenterAnnotator-v$version-win-x64-single.exe"
$validationRoot = Join-Path $repositoryRoot "build\release-validation"

$repositoryFull = [System.IO.Path]::GetFullPath($repositoryRoot).TrimEnd('\') + '\'
$validationFull = [System.IO.Path]::GetFullPath($validationRoot)
if (-not $validationFull.StartsWith($repositoryFull, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "拒绝使用仓库外验证目录：$validationFull"
}
if (Test-Path -LiteralPath $validationFull) {
    Remove-Item -LiteralPath $validationFull -Recurse -Force
}
New-Item -ItemType Directory -Path $validationFull | Out-Null

Expand-Archive -LiteralPath $zipPath -DestinationPath (Join-Path $validationFull "unzipped")
$portableExe = Get-ChildItem `
    -LiteralPath (Join-Path $validationFull "unzipped") `
    -Filter "AtomCenterAnnotator.exe" -Recurse |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $portableExe) {
    throw "ZIP 解压后未找到 AtomCenterAnnotator.exe"
}

$savedPath = $env:PATH
$savedPythonHome = $env:PYTHONHOME
$savedPythonPath = $env:PYTHONPATH
$env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

try {
    $portableReport = Join-Path $validationFull "portable-clean-env.json"
    $portableProcess = Start-Process `
        -FilePath $portableExe `
        -ArgumentList @("--smoke-test", $portableReport) `
        -WorkingDirectory $validationFull `
        -Wait -PassThru -WindowStyle Hidden
    if ($portableProcess.ExitCode -ne 0) {
        throw "ZIP 便携版清洁环境自检失败：$($portableProcess.ExitCode)"
    }
    $portableResult = Get-Content -LiteralPath $portableReport -Raw | ConvertFrom-Json
    if (
        -not $portableResult.ok `
        -or -not $portableResult.frozen `
        -or -not $portableResult.launcher_constructed `
        -or -not $portableResult.launcher_content_fits `
        -or -not $portableResult.launcher_buttons_present `
        -or -not $portableResult.sidebar_scrollable `
        -or -not $portableResult.sidebar_packed_first `
        -or -not $portableResult.sidebar_horizontal_fits `
        -or -not $portableResult.sidebar_vertical_overflow `
        -or -not $portableResult.image_switch_passed `
        -or -not $portableResult.bundled_font_registered
    ) {
        throw "ZIP 便携版自检断言失败：$portableReport"
    }

    $singleReport = Join-Path $validationFull "single-clean-env.json"
    $singleProcess = Start-Process `
        -FilePath $singlePath `
        -ArgumentList @("--smoke-test", $singleReport) `
        -WorkingDirectory $validationFull `
        -Wait -PassThru -WindowStyle Hidden
    if ($singleProcess.ExitCode -ne 0) {
        throw "单文件版清洁环境自检失败：$($singleProcess.ExitCode)"
    }
    $singleResult = Get-Content -LiteralPath $singleReport -Raw | ConvertFrom-Json
    if (
        -not $singleResult.ok `
        -or -not $singleResult.frozen `
        -or -not $singleResult.launcher_constructed `
        -or -not $singleResult.launcher_content_fits `
        -or -not $singleResult.launcher_buttons_present `
        -or -not $singleResult.sidebar_scrollable `
        -or -not $singleResult.sidebar_packed_first `
        -or -not $singleResult.sidebar_horizontal_fits `
        -or -not $singleResult.sidebar_vertical_overflow `
        -or -not $singleResult.image_switch_passed `
        -or -not $singleResult.bundled_font_registered
    ) {
        throw "单文件版自检断言失败：$singleReport"
    }
}
finally {
    $env:PATH = $savedPath
    if ($null -eq $savedPythonHome) {
        Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONHOME = $savedPythonHome
    }
    if ($null -eq $savedPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONPATH = $savedPythonPath
    }
}

[pscustomobject]@{
    ZipExtracted = $true
    PythonEnvironmentCleared = $true
    PortableFrozen = $portableResult.frozen
    PortableLauncher = $portableResult.launcher_constructed
    PortableLauncherFits = $portableResult.launcher_content_fits
    PortableButtonsPresent = $portableResult.launcher_buttons_present
    PortableSidebarScrollable = $portableResult.sidebar_scrollable
    PortableSidebarReserved = $portableResult.sidebar_packed_first
    PortableSidebarFits = $portableResult.sidebar_horizontal_fits
    PortableImageSwitch = $portableResult.image_switch_passed
    PortableBundledFont = $portableResult.bundled_font_registered
    PortableTIFFShape = ($portableResult.loaded_shape -join "x")
    PortableSavedPoints = $portableResult.saved_points
    PortableExportedPoints = $portableResult.exported_points
    SingleFileFrozen = $singleResult.frozen
    SingleFileLauncher = $singleResult.launcher_constructed
    SingleFileLauncherFits = $singleResult.launcher_content_fits
    SingleFileButtonsPresent = $singleResult.launcher_buttons_present
    SingleFileSidebarScrollable = $singleResult.sidebar_scrollable
    SingleFileSidebarReserved = $singleResult.sidebar_packed_first
    SingleFileSidebarFits = $singleResult.sidebar_horizontal_fits
    SingleFileImageSwitch = $singleResult.image_switch_passed
    SingleFileBundledFont = $singleResult.bundled_font_registered
} | Format-List
