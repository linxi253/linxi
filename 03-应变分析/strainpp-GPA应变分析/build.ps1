[CmdletBinding()]
param(
    [string]$CertificatePath = $env:STRAINPP_CERT_PATH,
    [string]$CertificatePassword = $env:STRAINPP_CERT_PASSWORD
)

# Native tools (python, PyInstaller, signtool) write progress to stderr; on
# Windows PowerShell 5.1 `$ErrorActionPreference = 'Stop'` turns those lines
# into terminating errors. Explicit $LASTEXITCODE checks below replace it.
$ErrorActionPreference = 'Continue'
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$specPath = Join-Path $projectDir 'Strain++GPA.spec'
$exePath = Join-Path $projectDir 'dist\Strain++GPA.exe'

# Version isolation: always build inside the project-local .venv, never with
# whatever "python" happens to be on PATH. Building against a shared
# interpreter makes the same source produce a different exe on every machine
# and lets missing dependencies leak into the global environment.
$venvPython = Join-Path $projectDir '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw @"
Project virtual environment not found:
  $venvPython

Create it once with:
  python -m venv .venv
  .venv\Scripts\python -X utf8 -m pip install -r requirements-lock.txt
"@
}

Push-Location $projectDir
try {
    & $venvPython -m unittest discover -s tests -v 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw 'Tests failed; executable was not rebuilt.'
    }

    & $venvPython -m PyInstaller --noconfirm --clean $specPath 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw 'PyInstaller build failed.'
    }

    if ($CertificatePath) {
        $resolvedCertificate = (Resolve-Path -LiteralPath $CertificatePath).Path
        # Sign from the user certificate store by thumbprint instead of
        # passing the PFX: signtool's /p flag puts the private-key password
        # on the process command line, where any same-machine process can
        # read it (WMI Win32_Process.CommandLine, ETW/Sysmon process-creation
        # events).  Importing with a SecureString keeps the password off the
        # command line entirely.
        if ($CertificatePassword) {
            $securePassword = ConvertTo-SecureString -String $CertificatePassword -AsPlainText -Force
        } else {
            $securePassword = New-Object System.Security.SecureString  # passwordless PFX
        }
        $importedCertificate = Import-PfxCertificate `
            -FilePath $resolvedCertificate `
            -CertStoreLocation Cert:\CurrentUser\My `
            -Password $securePassword
        $signArgs = @(
            'sign', '/fd', 'SHA256', '/td', 'SHA256',
            '/tr', 'http://timestamp.digicert.com',
            '/sha1', $importedCertificate.Thumbprint
        )
        $signArgs += $exePath
        & signtool.exe @signArgs 2>&1 | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw 'signtool failed.'
        }
        Write-Host "Built and signed: $exePath"
    } else {
        Write-Warning (
            'Built an unsigned executable. Set STRAINPP_CERT_PATH (and, if ' +
            'needed, STRAINPP_CERT_PASSWORD) to sign a release build.'
        )
        Write-Host "Built: $exePath"
    }
} finally {
    Pop-Location
}
