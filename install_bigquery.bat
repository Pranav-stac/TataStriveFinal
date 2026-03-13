@echo off
:: Install BigQuery dependency for TataStrive Analytics
:: Run this script once before launching the app.

echo.
echo  Installing google-cloud-bigquery and google-auth ...
echo  (This may take a few minutes)
echo.

pip install google-cloud-bigquery>=3.10.0 google-auth>=2.20.0 --no-deps-check

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  First attempt failed, trying with --no-cache-dir ...
    pip install google-cloud-bigquery google-auth --no-cache-dir
)

if %ERRORLEVEL% EQU 0 (
    echo.
    echo  ===================================================
    echo   BigQuery dependency installed successfully!
    echo   You can now launch TataStrive Analytics normally.
    echo  ===================================================
) else (
    echo.
    echo  ERROR: Installation failed.
    echo  Please check your internet connection and try again.
    echo  Or install manually:
    echo    pip install google-cloud-bigquery google-auth
)

echo.
pause
