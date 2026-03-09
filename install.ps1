# install.ps1 — Build and install llm-detector as a standalone .exe on Windows.
#
# Usage (PowerShell):
#   .\install.ps1                              # build + install to %LOCALAPPDATA%\llm-detector
#   .\install.ps1 -Prefix "C:\Tools"           # install to C:\Tools\bin
#   .\install.ps1 -BuildOnly                   # build without installing
#

param(
    [string]$Prefix = "$env:LOCALAPPDATA\llm-detector",
    [switch]$BuildOnly,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

if ($Help) {
    Write-Host @"
Usage: .\install.ps1 [-Prefix DIR] [-BuildOnly]

Options:
  -Prefix DIR    Install to DIR\bin (default: %LOCALAPPDATA%\llm-detector)
  -BuildOnly     Only build, do not install
"@
    exit 0
}

$InstallDir = Join-Path $Prefix "bin"
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "==> Checking Python environment..."
$pyVer = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
if (-not $pyVer -or [version]$pyVer -lt [version]"3.9") {
    Write-Error "Python 3.9 or later is required. Found: $pyVer"
    exit 1
}

Write-Host "==> Installing build dependencies..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet "$ScriptDir[bundle]"

Write-Host "==> Building single-file executable..."
Push-Location $ScriptDir
try {
    $env:ONEFILE = "1"
    pyinstaller llm_detector.spec --noconfirm --clean
} finally {
    Remove-Item Env:\ONEFILE -ErrorAction SilentlyContinue
    Pop-Location
}

$ExePath = Join-Path $ScriptDir "dist\llm-detector.exe"
if (-not (Test-Path $ExePath)) {
    Write-Error "Build failed - executable not found at $ExePath"
    exit 1
}

Write-Host "==> Build successful: $ExePath"

if ($BuildOnly) {
    Write-Host "Done (build only). Executable is at: $ExePath"
    exit 0
}

Write-Host "==> Installing to $InstallDir..."
New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
Copy-Item -Force $ExePath (Join-Path $InstallDir "llm-detector.exe")

# Check if install dir is on PATH
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($currentPath -notlike "*$InstallDir*") {
    Write-Host ""
    Write-Host "NOTE: $InstallDir is not on your PATH."
    Write-Host "Adding it now for the current user..."
    [Environment]::SetEnvironmentVariable(
        "PATH",
        "$InstallDir;$currentPath",
        "User"
    )
    $env:PATH = "$InstallDir;$env:PATH"
    Write-Host "Done. Restart your terminal for the change to take effect."
}

Write-Host ""
Write-Host "==> Installed successfully!"
Write-Host "    Run with:  llm-detector --help"
