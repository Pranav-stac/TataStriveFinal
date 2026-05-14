param(
    [Parameter(Mandatory = $true)]
    [string]$ReleaseZipUrl,

    [string]$InstallDir = "$env:LOCALAPPDATA\TataStriveAnalytics",
    [switch]$NoShortcut,
    [switch]$NoLaunch
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[Setup] $Message"
}

function New-Shortcut {
    param(
        [Parameter(Mandatory = $true)][string]$ShortcutPath,
        [Parameter(Mandatory = $true)][string]$TargetPath,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($ShortcutPath)
    $shortcut.TargetPath = $TargetPath
    $shortcut.WorkingDirectory = $WorkingDirectory
    $shortcut.IconLocation = "$TargetPath,0"
    $shortcut.Save()
}

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
} catch {
    # Ignore if the runtime already uses modern TLS defaults.
}

$tempRoot = Join-Path $env:TEMP ("TataStriveSetup_" + [guid]::NewGuid().ToString("N"))
$zipPath = Join-Path $tempRoot "TataStriveAnalytics.zip"
$extractPath = Join-Path $tempRoot "extract"

try {
    Write-Step "Preparing temporary workspace..."
    New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $extractPath -Force | Out-Null

    Write-Step "Downloading release package..."
    Invoke-WebRequest -Uri $ReleaseZipUrl -OutFile $zipPath

    Write-Step "Extracting package..."
    Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force

    # Support either:
    # 1) ZIP root contains TataStriveAnalytics.exe directly
    # 2) ZIP root contains a single folder with the executable
    $rootExe = Join-Path $extractPath "TataStriveAnalytics.exe"
    if (Test-Path $rootExe) {
        $sourceDir = $extractPath
    } else {
        $exeCandidate = Get-ChildItem -Path $extractPath -Recurse -File -Filter "TataStriveAnalytics.exe" |
            Select-Object -First 1
        if (-not $exeCandidate) {
            throw "TataStriveAnalytics.exe not found in downloaded package."
        }
        $sourceDir = $exeCandidate.Directory.FullName
    }

    if (Test-Path $InstallDir) {
        Write-Step "Removing previous installation..."
        Remove-Item -Path $InstallDir -Recurse -Force
    }

    Write-Step "Installing to: $InstallDir"
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    Copy-Item -Path (Join-Path $sourceDir "*") -Destination $InstallDir -Recurse -Force

    $exePath = Join-Path $InstallDir "TataStriveAnalytics.exe"
    if (-not (Test-Path $exePath)) {
        throw "Installation failed: executable not found at $exePath"
    }

    $vcInstaller = Join-Path $InstallDir "vc_redist.x64.exe"
    $vcInstalled = $false
    foreach ($key in @(
        "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
    )) {
        try {
            if ((Get-ItemProperty -Path $key -Name Installed -ErrorAction Stop).Installed -eq 1) {
                $vcInstalled = $true
                break
            }
        } catch {
        }
    }

    if (-not $vcInstalled -and (Test-Path $vcInstaller)) {
        Write-Step "Installing Microsoft Visual C++ 2015-2022 Redistributable (x64)..."
        Start-Process -FilePath $vcInstaller -ArgumentList "/install", "/passive", "/norestart" -Verb RunAs -Wait
    }

    if (-not $NoShortcut) {
        Write-Step "Creating shortcuts..."
        $desktopPath = [Environment]::GetFolderPath("Desktop")
        $startMenuPrograms = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"

        $desktopShortcut = Join-Path $desktopPath "TataStrive Analytics.lnk"
        $startMenuShortcut = Join-Path $startMenuPrograms "TataStrive Analytics.lnk"

        New-Shortcut -ShortcutPath $desktopShortcut -TargetPath $exePath -WorkingDirectory $InstallDir
        New-Shortcut -ShortcutPath $startMenuShortcut -TargetPath $exePath -WorkingDirectory $InstallDir
    }

    Write-Host ""
    Write-Host "TataStrive Analytics installation completed successfully."
    Write-Host "Installed at: $InstallDir"
    Write-Host ""

    if (-not $NoLaunch) {
        Write-Step "Launching application..."
        Start-Process -FilePath $exePath -WorkingDirectory $InstallDir
    }
}
finally {
    if (Test-Path $tempRoot) {
        Remove-Item -Path $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
