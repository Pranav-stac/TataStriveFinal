# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs

datas = [('E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\app\\resources', 'resources'), ('E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\app', 'app'), ('E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\classroom_analysis', 'classroom_analysis'), ('E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\Models', 'Models'), ('E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\.env', '.'), ('E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\yolov8n.pt', '.'), ('E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\yolov8m.pt', '.'), ('E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\yolov8n-pose.pt', '.'), ('E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\osnet_x1_0_msmt17.pt', '.')]
binaries = [('E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\env\\Lib\\site-packages\\onnxruntime\\capi\\onnxruntime.dll', 'onnxruntime/capi'), ('E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\env\\Lib\\site-packages\\onnxruntime\\capi\\onnxruntime_providers_cuda.dll', 'onnxruntime/capi'), ('E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\env\\Lib\\site-packages\\onnxruntime\\capi\\onnxruntime_providers_shared.dll', 'onnxruntime/capi'), ('E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\env\\Lib\\site-packages\\onnxruntime\\capi\\onnxruntime_providers_tensorrt.dll', 'onnxruntime/capi'), ('E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\env\\Lib\\site-packages\\onnxruntime\\capi\\onnxruntime_pybind11_state.pyd', 'onnxruntime/capi')]
datas += collect_data_files('boxmot')
datas += collect_data_files('insightface')
binaries += collect_dynamic_libs('onnxruntime')


a = Analysis(
    ['E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\app\\main.py'],
    pathex=['E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\env'],
    binaries=binaries,
    datas=datas,
    hiddenimports=['typing_extensions', 'ultralytics', 'ultralytics.models', 'ultralytics.nn', 'ultralytics.utils', 'onnx', 'omegaconf', 'boxmot', 'insightface', 'onnxruntime', 'scipy.spatial.distance', 'cv2', 'torch', 'torchvision', 'numpy', 'PIL', 'groq', 'dotenv', 'PyQt6', 'PyQt6.QtWidgets', 'PyQt6.QtCore', 'PyQt6.QtGui'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\pyi_rth_onnxruntime.py'],
    excludes=['onnx.reference', 'pytest'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TataStriveAnalytics',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='TataStriveAnalytics',
)
