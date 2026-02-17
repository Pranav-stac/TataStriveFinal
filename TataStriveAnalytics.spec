# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\app\\resources', 'resources'), ('E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\app', 'app'), ('E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\classroom_analysis', 'classroom_analysis'), ('E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\Models', 'Models')]
binaries = []
hiddenimports = ['ultralytics', 'typing_extensions', 'omegaconf', 'boxmot', 'insightface', 'onnxruntime', 'scipy.spatial.distance', 'cv2', 'torch', 'torchvision', 'numpy', 'PIL', 'groq', 'python-dotenv', 'dotenv', 'PyQt6', 'PyQt6.QtWidgets', 'PyQt6.QtCore', 'PyQt6.QtGui']
tmp_ret = collect_all('ultralytics')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('insightface')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('onnxruntime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['E:\\Pranav\\InternshipFreelancing\\TataStriveFinal\\app\\main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name='TataStriveAnalytics',
)
