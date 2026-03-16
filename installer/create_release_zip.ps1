param(
    [string]$DistDir = ".\dist\TataStriveAnalytics",
    [string]$OutDir = ".\release",
    [string]$Version = "latest"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $DistDir)) {
    throw "Dist folder not found: $DistDir. Build first using: python build_exe.py"
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$resolvedDist = (Resolve-Path $DistDir).Path
$resolvedOut = (Resolve-Path $OutDir).Path

$zipName = "TataStriveAnalytics-$Version-win64.zip"
$zipPath = Join-Path $resolvedOut $zipName

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

# Zip directly (close other apps using dist folder if you get "file in use" errors)
Compress-Archive -Path (Join-Path $resolvedDist "*") -DestinationPath $zipPath -CompressionLevel Optimal

Write-Host "Release ZIP created:"
Write-Host $zipPath
