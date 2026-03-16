param(
    [string]$Version = "1.0.0"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$distDir = Join-Path $projectRoot "dist\TataStriveAnalytics"
$issPath = Join-Path $PSScriptRoot "TataStriveAnalytics.iss"
$releaseDir = Join-Path $projectRoot "release"

if (-not (Test-Path $distDir)) {
    throw "Build output not found at $distDir. Run 'python build_exe.py' first."
}

if (-not (Test-Path $issPath)) {
    throw "Inno Setup script not found: $issPath"
}

$isccCandidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles}\Inno Setup 6\ISCC.exe"
)

$iscc = $isccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) {
    throw "ISCC.exe not found. Install Inno Setup 6 from https://jrsoftware.org/isdl.php"
}

New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null

& $iscc "/DMyAppVersion=$Version" $issPath

Write-Host ""
Write-Host "Installer build complete."
Write-Host "Output folder: $releaseDir"
