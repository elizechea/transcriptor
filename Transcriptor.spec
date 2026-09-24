# -*- mode: python ; coding: utf-8 -*-
# pyinstaller --noconfirm Transcriptor.spec
#   Windows -> dist/Transcriptor.exe (un único archivo)
#   macOS   -> dist/Transcriptor.app (Apple Silicon)
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

VERSION = "1.1.0"
IS_MAC = sys.platform == "darwin"

datas = [("assets/icon.ico", "assets"), ("assets/icon.png", "assets")]
datas += collect_data_files("faster_whisper")  # modelo VAD silero (.onnx)
binaries = collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("onnxruntime")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=["transcriptor.selftest"] + ([] if IS_MAC else ["windnd"]),
    excludes=["torch", "tensorflow", "matplotlib", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

if IS_MAC:
    # En macOS el formato adecuado es un .app (carpeta), no un ejecutable onefile.
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="Transcriptor",
        console=False,
        upx=False,
        strip=False,
        debug=False,
        target_arch="arm64",
    )
    coll = COLLECT(exe, a.binaries, a.datas, name="Transcriptor", upx=False, strip=False)
    app = BUNDLE(
        coll,
        name="Transcriptor.app",
        icon="assets/icon.icns",
        bundle_identifier="app.transcriptor.desktop",
        version=VERSION,
        info_plist={
            "CFBundleDisplayName": "Transcriptor",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "LSMinimumSystemVersion": "13.0",
            "LSApplicationCategoryType": "public.app-category.productivity",
            "NSHighResolutionCapable": True,
            # La interfaz usa colores claros fijos: se evita el modo oscuro.
            "NSRequiresAquaSystemAppearance": True,
            "NSDownloadsFolderUsageDescription":
                "Transcriptor guarda las transcripciones en la carpeta Descargas.",
            "NSDesktopFolderUsageDescription":
                "Transcriptor necesita leer los audios y vídeos que elijas.",
            "NSDocumentsFolderUsageDescription":
                "Transcriptor necesita leer los audios y vídeos que elijas.",
            "NSRemovableVolumesUsageDescription":
                "Transcriptor necesita leer los audios y vídeos que elijas.",
            "NSNetworkVolumesUsageDescription":
                "Transcriptor necesita leer los audios y vídeos que elijas.",
        },
    )
else:
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
