# -*- mode: python ; coding: utf-8 -*-
# pyinstaller --noconfirm Transcriptor.spec
#   Windows -> dist/Transcriptor.exe (un único archivo)
#   macOS   -> dist/Transcriptor.app (Apple Silicon)
#   Linux   -> dist/Transcriptor (un único archivo)
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

VERSION = "1.5.0"
IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"

datas = [("assets/icon.ico", "assets"), ("assets/icon.png", "assets")]
datas += collect_data_files("faster_whisper")  # modelo VAD silero (.onnx)
binaries = collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("onnxruntime")
if IS_MAC:
    # MLX (GPU de Apple): libmlx.dylib, los kernels Metal (mlx.metallib) y los
    # filtros mel y tokenizadores de mlx-whisper.
    binaries += collect_dynamic_libs("mlx")
    datas += collect_data_files("mlx", includes=["lib/*.metallib"])
    datas += collect_data_files("mlx_whisper")
    hiddenimports = collect_submodules("mlx")
    # numba/llvmlite y scipy solo sirven para marcas por palabra (ver mlx_backend).
    excludes = ["numba", "llvmlite", "scipy"]
else:
    hiddenimports = ["windnd"] if IS_WIN else []
    excludes = []

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=["transcriptor.selftest"] + hiddenimports,
    excludes=["torch", "tensorflow", "matplotlib", "IPython", "pytest"] + excludes,
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
            # Sin App Nap: la transcripción no se frena con la ventana oculta.
            "NSAppSleepDisabled": True,
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
        icon="assets/icon.ico" if IS_WIN else None,
        console=False,
        upx=False,
        strip=False,
        debug=False,
        version="version_info.txt" if IS_WIN else None,
    )
