# -*- mode: python ; coding: utf-8 -*-
# Genera un único Transcriptor.exe:  pyinstaller --noconfirm Transcriptor.spec
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = [("assets/icon.ico", "assets"), ("assets/icon.png", "assets")]
datas += collect_data_files("faster_whisper")  # modelo VAD silero (.onnx)
binaries = collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("onnxruntime")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=["windnd", "transcriptor.selftest"],
    excludes=["torch", "tensorflow", "matplotlib", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Transcriptor",
    icon="assets/icon.ico",
    console=False,
    upx=False,
    strip=False,
    debug=False,
    version="version_info.txt",
)
