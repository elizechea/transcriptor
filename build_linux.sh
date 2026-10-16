#!/bin/bash
# Compila dist/Transcriptor y dist/Transcriptor-linux-<arquitectura>.tar.gz en Linux.
#
# Con uv (recomendado, https://docs.astral.sh/uv/) se descarga un Python 3.12 que
# ya incluye Tk. Sin uv hace falta Python 3.12 con venv y tkinter del sistema
# (Debian/Ubuntu: sudo apt install python3.12-venv python3-tk).
#
# El ejecutable solo funciona en distribuciones con una glibc igual o más nueva que
# la del equipo donde se compila: para repartirlo, compila en una distribución antigua.
set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null; then
    [ -d .venv ] || uv venv --python 3.12 .venv
    source .venv/bin/activate
    uv pip install -r requirements-build.txt
else
    PY=${PYTHON:-python3.12}
    [ -d .venv ] || "$PY" -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements-build.txt
fi

python -m pytest -q tests
pyinstaller --noconfirm --clean Transcriptor.spec
tar -czf "dist/Transcriptor-linux-$(uname -m).tar.gz" -C dist Transcriptor
echo "Listo: dist/Transcriptor y dist/Transcriptor-linux-$(uname -m).tar.gz"
