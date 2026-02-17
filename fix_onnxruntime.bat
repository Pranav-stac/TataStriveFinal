@echo off
REM Fix onnxruntime DLL errors (common after building .exe or when face matching fails)
REM Use CPU-only onnxruntime for reliable builds and distribution
echo Uninstalling onnxruntime-gpu (if present)...
pip uninstall onnxruntime-gpu -y 2>nul
echo Installing onnxruntime (CPU)...
pip install onnxruntime
echo.
echo Done. Rebuild with: python build_exe.py
pause
