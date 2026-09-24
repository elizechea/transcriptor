"""Rutas del sistema: carpeta de Descargas, datos de la aplicación, nombres de salida."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from . import APP_NAME

# GUID de la carpeta conocida "Descargas" en Windows (FOLDERID_Downloads).
_FOLDERID_DOWNLOADS = "{374DE290-123F-4565-9164-39C4925E467B}"


def _windows_downloads() -> Path | None:
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", wintypes.BYTE * 8),
        ]

    guid = GUID()
    ctypes.oledll.ole32.CLSIDFromString(_FOLDERID_DOWNLOADS, ctypes.byref(guid))
    path_ptr = ctypes.c_wchar_p()
    try:
        ctypes.windll.shell32.SHGetKnownFolderPath(
            ctypes.byref(guid), 0, None, ctypes.byref(path_ptr)
        )
        return Path(path_ptr.value) if path_ptr.value else None
    finally:
        ctypes.windll.ole32.CoTaskMemFree(path_ptr)


def downloads_dir() -> Path:
    """Devuelve la carpeta de Descargas del usuario (respetando si la ha movido)."""
    candidate: Path | None = None
    try:
        if sys.platform == "win32":
            candidate = _windows_downloads()
        elif sys.platform.startswith("linux"):
            out = subprocess.run(
                ["xdg-user-dir", "DOWNLOAD"], capture_output=True, text=True, timeout=2
            ).stdout.strip()
            if out and Path(out) != Path.home():
                candidate = Path(out)
    except Exception:
        candidate = None
    if candidate is None:
        candidate = Path.home() / "Downloads"
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def app_data_dir() -> Path:
    """Carpeta para modelos, ajustes y registros (%LOCALAPPDATA%\\Transcriptor)."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    path = base / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = app_data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_file() -> Path:
    return app_data_dir() / "transcriptor.log"


_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def output_path_for(source: Path, out_dir: Path) -> Path:
    """Nombre del .md de salida sin sobrescribir transcripciones anteriores."""
    stem = _INVALID_CHARS.sub("_", source.stem).strip(" .") or "transcripcion"
    target = out_dir / f"{stem} - transcripción.md"
    n = 2
    while target.exists():
        target = out_dir / f"{stem} - transcripción ({n}).md"
        n += 1
    return target


def load_settings() -> dict:
    try:
        return json.loads((app_data_dir() / "settings.json").read_text("utf-8"))
    except Exception:
        return {}


def save_settings(settings: dict) -> None:
    try:
        (app_data_dir() / "settings.json").write_text(
            json.dumps(settings, ensure_ascii=False, indent=2), "utf-8"
        )
    except Exception:
        pass


def open_in_explorer(path: Path, select: bool = False) -> None:
    """Abre una carpeta o archivo con la aplicación predeterminada del sistema."""
    if sys.platform == "win32":
        if select and path.is_file():
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)] if select else ["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
