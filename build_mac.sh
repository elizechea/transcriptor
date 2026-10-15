#!/bin/bash
# Compila Transcriptor.app y Transcriptor-macOS.dmg en un Mac con chip M.
# Requiere Python 3.12 de python.org (incluye Tk): https://www.python.org/downloads/macos/
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3.12}
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-build.txt
# pip elige la rueda de MLX para la versión de macOS de este equipo; la de
# macOS 14 funciona en 14, 15 y 26 (la app exige 14 para usar la GPU).
MLX=$(sed -n 's/^mlx==\([^;]*\).*/\1/p' requirements.txt)
rm -rf build/mlx-wheels
pip download --quiet --only-binary=:all: --platform macosx_14_0_arm64 --no-deps \
  -d build/mlx-wheels "mlx==$MLX" "mlx-metal==$MLX"
pip install --force-reinstall --no-deps build/mlx-wheels/*.whl
python -m pytest -q tests
pyinstaller --noconfirm --clean Transcriptor.spec
codesign --force --deep --sign - dist/Transcriptor.app
rm -rf dmg && mkdir dmg && cp -R dist/Transcriptor.app dmg/ && ln -s /Applications dmg/Aplicaciones
hdiutil create -volname Transcriptor -srcfolder dmg -ov -format UDZO dist/Transcriptor-macOS.dmg
rm -rf dmg
echo "Listo: dist/Transcriptor.app y dist/Transcriptor-macOS.dmg"
