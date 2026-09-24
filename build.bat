@echo off
rem Compila Transcriptor.exe en Windows (requiere Python 3.10-3.12 de python.org).
setlocal
cd /d "%~dp0"
if not exist .venv (
    py -3.12 -m venv .venv || python -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat || goto :error
python -m pip install --upgrade pip || goto :error
pip install -r requirements-build.txt || goto :error
python -m pytest -q tests || goto :error
pyinstaller --noconfirm --clean Transcriptor.spec || goto :error
echo.
echo Listo: dist\Transcriptor.exe
exit /b 0
:error
echo.
echo La compilacion ha fallado.
exit /b 1
