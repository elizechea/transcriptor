"""Punto de entrada de Transcriptor (también el del .exe generado con PyInstaller)."""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys

# Descargas por HTTP clásico (progreso medible) y sin avisos de symlinks en Windows.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Evita el fallo "OMP: Error #15" cuando se cargan dos copias de OpenMP.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# En un .exe sin consola sys.stdout/sys.stderr son None y cualquier print o
# barra de tqdm de una librería provocaría una excepción.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _setup_logging() -> None:
    from transcriptor.paths import log_file

    handlers: list[logging.Handler] = []
    try:
        path = log_file()
        if path.exists() and path.stat().st_size > 5_000_000:
            path.unlink()
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    except Exception:
        pass
    handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def _enable_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def main() -> int:
    multiprocessing.freeze_support()
    _setup_logging()
    if "--self-test" in sys.argv:
        from transcriptor.selftest import run_self_test

        return run_self_test(sys.argv)
    _enable_dpi_awareness()
    from transcriptor.gui import run

    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
