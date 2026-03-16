# Windows Double-Click Installer

This project includes an Inno Setup configuration that creates a single installer file:

`TataStriveAnalytics_Setup_<version>.exe`

Your client only needs to download that file and double-click it.

## 1) Build the compiled app

From project root:

```powershell
python build_exe.py
```

This creates the compiled app in `dist/TataStriveAnalytics/`.

## 2) Install Inno Setup (build machine only)

Install **Inno Setup 6**:

[https://jrsoftware.org/isdl.php](https://jrsoftware.org/isdl.php)

## 3) Build installer EXE

```powershell
powershell -ExecutionPolicy Bypass -File .\installer\build_installer.ps1 -Version 1.0.0
```

Output:

`release/TataStriveAnalytics_Setup_1.0.0.exe`

## 4) Share with client

Upload/share that installer EXE (Google Drive, OneDrive, website, etc.).

Client workflow:
- Download `TataStriveAnalytics_Setup_1.0.0.exe`
- Double-click installer
- Finish wizard
- Launch from Desktop/Start Menu shortcut

## Included files

The installer automatically bundles everything from:

`dist/TataStriveAnalytics/`

That means the client does not need Python, pip, or any scripts.
