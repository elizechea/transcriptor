"""Actualización automática desde las releases de GitHub.

Al arrancar se consulta la última release; si es más nueva y el usuario acepta,
se descarga el archivo de este sistema, se comprueba su huella SHA-256, se
sustituye el programa instalado y se vuelve a abrir.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable

import certifi

from . import __version__
from .paths import system_env

log = logging.getLogger(__name__)

REPO = "elizechea/transcriptor"
RELEASES_URL = f"https://github.com/{REPO}/releases/latest"
_API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
_HEADERS = {"User-Agent": f"Transcriptor/{__version__}",
            "Accept": "application/vnd.github+json"}


# Los certificados de certifi (los mismos que usa la descarga de modelos): el Python
# empaquetado no siempre encuentra los del sistema.
_SSL = ssl.create_default_context(cafile=certifi.where())


def _open(url: str, timeout: int):
    request = urllib.request.Request(url, headers=_HEADERS)
    return urllib.request.urlopen(request, timeout=timeout, context=_SSL)


class UpdateError(Exception):
    pass


def parse_version(tag: str) -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in tag.strip().lstrip("vV").split("."))
    except ValueError:
        return ()


def asset_name() -> str | None:
    if sys.platform == "win32":
        return "Transcriptor.exe"
    if sys.platform == "darwin":
        return "Transcriptor-macOS.dmg"
    if sys.platform.startswith("linux") and platform.machine() == "x86_64":
        return "Transcriptor-linux-x86_64.tar.gz"
    return None


def installed_path() -> Path:
    """El .exe, el ejecutable de Linux o el Transcriptor.app en ejecución."""
    exe = Path(sys.executable).resolve()
    return exe.parents[2] if sys.platform == "darwin" else exe


def can_update() -> bool:
    return bool(getattr(sys, "frozen", False)) and asset_name() is not None


def find_update(release: dict, current: str = __version__) -> dict | None:
    """El asset de este sistema si `release` es más nueva que `current`."""
    if release.get("draft") or release.get("prerelease"):
        return None
    tag = release.get("tag_name", "")
    if parse_version(tag) <= parse_version(current):
        return None
    for asset in release.get("assets", []):
        if asset.get("name") == asset_name():
            return {"version": tag.lstrip("vV"), "url": asset["browser_download_url"],
                    "size": asset.get("size", 0), "digest": asset.get("digest") or ""}
    return None


def check() -> dict | None:
    with _open(_API_URL, timeout=10) as response:
        return find_update(json.load(response))


def cleanup_old() -> None:
    """Borra el .exe anterior que dejó una actualización en Windows."""
    if sys.platform == "win32" and getattr(sys, "frozen", False):
        Path(sys.executable).with_suffix(".old.exe").unlink(missing_ok=True)


def download(update: dict, dest: Path, progress: Callable[[int, int], None]) -> None:
    """Descarga el asset en `dest` y comprueba su huella; si no coincide, lo borra."""
    algo, _, expected = update["digest"].partition(":")
    if algo != "sha256" or not expected:
        raise UpdateError("La release no publica la huella SHA-256 del archivo.")
    digest = hashlib.sha256()
    done = 0
    try:
        with _open(update["url"], timeout=30) as response, open(dest, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                progress(done, update["size"])
        if digest.hexdigest() != expected.lower():
            raise UpdateError("El archivo descargado está dañado (la huella no coincide).")
    except BaseException:
        dest.unlink(missing_ok=True)
        raise


def install(update: dict, progress: Callable[[int, int], None]) -> Path:
    """Descarga y sustituye el programa instalado. Devuelve lo que hay que abrir."""
    target = installed_path()
    if sys.platform == "darwin":
        if "/AppTranslocation/" in str(target) or str(target).startswith("/Volumes/"):
            raise UpdateError("Mueve Transcriptor a la carpeta Aplicaciones y vuelve a abrirlo "
                              "para poder actualizarlo.")
        with tempfile.TemporaryDirectory() as tmp:
            dmg = Path(tmp) / "Transcriptor.dmg"
            download(update, dmg, progress)
            _replace_app(dmg, target, Path(tmp) / "mnt")
        return target
    # Archivo temporal junto al destino: os.replace solo es atómico en el mismo disco.
    new = target.with_name(f".{target.name}.new")
    try:
        if sys.platform == "win32":
            download(update, new, progress)
            # Windows no deja sobrescribir un .exe abierto, pero sí renombrarlo.
            old = target.with_suffix(".old.exe")
            old.unlink(missing_ok=True)
            os.replace(target, old)
            try:
                os.replace(new, target)
            except OSError:
                os.replace(old, target)
                raise
        else:
            archive = target.with_name(f".{target.name}.tar.gz")
            try:
                download(update, archive, progress)
                with tarfile.open(archive) as tar, open(new, "wb") as out:
                    member = tar.extractfile("Transcriptor")
                    if member is None:
                        raise UpdateError("El archivo descargado no contiene el programa.")
                    shutil.copyfileobj(member, out)
            finally:
                archive.unlink(missing_ok=True)
            new.chmod(0o755)
            os.replace(new, target)  # el proceso en marcha conserva el archivo anterior
    except PermissionError as exc:
        raise UpdateError(f"No hay permiso para escribir en {target.parent}.") from exc
    finally:
        new.unlink(missing_ok=True)
    return target


def _replace_app(dmg: Path, app: Path, mountpoint: Path) -> None:
    mountpoint.mkdir()
    subprocess.run(["hdiutil", "attach", "-nobrowse", "-readonly", "-mountpoint",
                    str(mountpoint), str(dmg)], check=True, capture_output=True)
    new = app.with_name(f".{app.name}.new")
    old = app.with_name(f".{app.name}.old")
    try:
        for leftover in (new, old):
            shutil.rmtree(leftover, ignore_errors=True)
        # ditto conserva la firma y los enlaces simbólicos del paquete.
        subprocess.run(["ditto", str(mountpoint / "Transcriptor.app"), str(new)],
                       check=True, capture_output=True)
        os.replace(app, old)
        try:
            os.replace(new, app)
        except OSError:
            os.replace(old, app)
            raise
        shutil.rmtree(old, ignore_errors=True)
    except PermissionError as exc:
        raise UpdateError(f"No hay permiso para escribir en {app.parent}.") from exc
    except subprocess.CalledProcessError as exc:
        raise UpdateError(f"No se pudo copiar la nueva versión: {exc.stderr!r}") from exc
    finally:
        shutil.rmtree(new, ignore_errors=True)
        subprocess.run(["hdiutil", "detach", "-force", str(mountpoint)], capture_output=True)


def open_releases_page() -> None:
    if sys.platform.startswith("linux"):
        # webbrowser heredaría las bibliotecas de PyInstaller (ver system_env).
        subprocess.Popen(["xdg-open", RELEASES_URL], env=system_env())
    else:
        import webbrowser
        webbrowser.open(RELEASES_URL)


def relaunch(target: Path) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-n", str(target)])
        return
    env = system_env() or dict(os.environ)
    # Que el nuevo ejecutable de PyInstaller no reutilice la carpeta temporal de este.
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    subprocess.Popen([str(target)], env=env, cwd=str(target.parent))
