# Manual delta update for TataStriveAnalytics (when in-app updater does not run).
# Run from the folder that contains TataStriveAnalytics.exe:
#   powershell -ExecutionPolicy Bypass -File "C:\path\to\apply_manual_update.ps1"
param(
    [string]$Repo = "Pranav-stac/TataStriveFinal"
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
if (Test-Path (Join-Path $Root "TataStriveAnalytics.exe")) {
    $InstallDir = $Root
} elseif (Test-Path (Join-Path $Root "..\TataStriveAnalytics\TataStriveAnalytics.exe")) {
    $InstallDir = (Resolve-Path (Join-Path $Root "..\TataStriveAnalytics")).Path
} else {
    Write-Host "Place this script in the TataStriveAnalytics folder (next to the .exe)."
    exit 1
}

Write-Host "Install dir: $InstallDir"
Set-Location $InstallDir

$release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" -Headers @{ "User-Agent" = "TataStriveManualUpdate" }
$tag = $release.tag_name
Write-Host "Latest release: $tag"

$patchUrl = ($release.assets | Where-Object { $_.name -eq "patch.zip" }).browser_download_url
if (-not $patchUrl) {
    Write-Host "ERROR: patch.zip not found on release $tag"
    exit 1
}

$zip = Join-Path $env:TEMP "tatastrive_patch.zip"
Invoke-WebRequest -Uri $patchUrl -OutFile $zip -UseBasicParsing
Write-Host "Downloaded patch.zip"

Expand-Archive -Path $zip -DestinationPath $InstallDir -Force
Remove-Item $zip -Force

Get-ChildItem -Path $InstallDir -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "Patch applied. Restart TataStriveAnalytics.exe (or Run_TataStrive.bat)."
Write-Host "Check app\__init__.py for __version__ and update.log for updater history."
