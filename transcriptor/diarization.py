"""Identificación de hablantes (opcional) con NVIDIA Nemotron 3 Diarization.

El modelo (107 MB, licencia OpenMDW) se ejecuta con NeMo-Speech.cpp, un programa
nativo de ~5 MB (Apache 2.0). Ninguno de los dos va dentro de la app: se descargan
la primera vez que se activa la opción y se comprueba su huella SHA-256.

Se usa el preset «v3-offline»: con el de streaming por defecto, en una grabación
de casi 3 horas las etiquetas de dos personas se intercambiaban a mitad.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import audio
from .paths import models_dir, system_env

log = logging.getLogger(__name__)

RUNTIME_VERSION = "0.2.0"
_RUNTIME_URL = ("https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/"
                f"v{RUNTIME_VERSION}/")
# (sistema, arquitectura) -> (archivo, SHA-256, tamaño). En el Mac, la versión Metal
# también funciona con la CPU.
_RUNTIMES = {
    ("win32", "AMD64"): (
        f"nemo-speech-{RUNTIME_VERSION}-windows-x86_64-cpu.zip",
        "82c80451086e86194aba9af19f800da77cd66d1c28d993d0efceffd6cdb49bb8", 5_528_418),
    ("darwin", "arm64"): (
        f"nemo-speech-{RUNTIME_VERSION}-macos-aarch64-metal.tar.gz",
        "5cb02ba7c04f0b5585ce5cde9c830be5f0b7dfb4c83083c500109c381b4f2da9", 5_031_510),
    ("linux", "x86_64"): (
        f"nemo-speech-{RUNTIME_VERSION}-linux-x86_64-cpu.tar.gz",
        "f396057150f1b774935c7414fd32ebc04195a795a34120d78d8c9ecefa1b7507", 6_683_477),
}
MODEL_FILE = "Nemotron-3-Diarization.q8_0.gguf"
_MODEL_URL = ("https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/"
              f"f667ed73aee57d40cc39428eb768b4fd87a0a29e/{MODEL_FILE}")
_MODEL_SHA256 = "08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1"
_MODEL_SIZE = 107_012_128

Emit = Callable[..., None]


@dataclass
class Turn:
    start: float
    end: float
    speaker: int


def _runtime() -> tuple[str, str, int] | None:
    system = "linux" if sys.platform.startswith("linux") else sys.platform
    return _RUNTIMES.get((system, platform.machine()))


def supported() -> bool:
    return _runtime() is not None


def _fetch(url: str, sha256: str, size: int, dest: Path, what: str, emit: Emit,
           check_cancel: Callable[[], None]) -> None:
    from .updater import download

    def progress(done: int, total: int) -> None:
        check_cancel()  # download() borra el archivo a medias al interrumpirse
        emit("download", done=done, total=total)

    emit("status", text=f"Descargando {what} (solo la primera vez)…")
    try:
        download({"url": url, "size": size, "digest": f"sha256:{sha256}"}, dest, progress)
    except OSError as exc:
        raise RuntimeError(f"No se pudo descargar {what}. Comprueba la conexión a "
                           f"Internet.\n\nDetalle: {exc}") from exc


def ensure_runtime(emit: Emit, check_cancel: Callable[[], None]) -> tuple[Path, Path]:
    """Rutas del programa nemo-speech y del modelo, descargándolos si hace falta."""
    runtime = _runtime()
    if runtime is None:
        raise RuntimeError("La identificación de hablantes no está disponible en este sistema.")
    name, sha256, size = runtime
    base = models_dir()
    folder = base / f"nemo-speech-{RUNTIME_VERSION}"
    exe = folder / "bin" / ("nemo-speech.exe" if sys.platform == "win32" else "nemo-speech")
    if not exe.exists():
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            archive = Path(tmp) / name
            _fetch(_RUNTIME_URL + name, sha256, size, archive,
                   "el programa de identificación de hablantes", emit, check_cancel)
            if name.endswith(".zip"):
                with zipfile.ZipFile(archive) as z:
                    z.extractall(tmp)
            else:
                with tarfile.open(archive) as t:
                    t.extractall(tmp, filter="data")  # conserva los enlaces de las .dylib/.so
            shutil.rmtree(folder, ignore_errors=True)
            os.replace(Path(tmp) / name.removesuffix(".zip").removesuffix(".tar.gz"), folder)
    model = base / MODEL_FILE
    if not model.exists():
        partial = model.with_name(model.name + ".part")
        _fetch(_MODEL_URL, _MODEL_SHA256, _MODEL_SIZE, partial,
               "el modelo de identificación de hablantes", emit, check_cancel)
        os.replace(partial, model)
    return exe, model


def _write_wav(source: Path, dest: Path, check_cancel: Callable[[], None]) -> None:
    """nemo-speech solo lee WAV: se decodifica a mono 16 kHz por bloques."""
    with wave.open(str(dest), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(audio.SAMPLE_RATE)
        for chunk in audio.iter_audio_chunks(source):
            check_cancel()
            w.writeframes((np.clip(chunk.samples, -1, 1) * 32767).astype("<i2").tobytes())


def _run(cmd: list[str], check_cancel: Callable[[], None]) -> None:
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err,
                                env=system_env(), creationflags=flags)
        try:
            while True:
                try:
                    proc.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    check_cancel()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        if proc.returncode != 0:
            err.seek(0)
            detail = err.read().decode("utf-8", "replace").strip().splitlines()[-5:]
            raise RuntimeError(f"nemo-speech terminó con código {proc.returncode}: "
                               + " ".join(detail))


def diarize(source: Path, emit: Emit, check_cancel: Callable[[], None]) -> list[Turn]:
    """Tramos de cada hablante en `source`, ordenados por inicio."""
    exe, model = ensure_runtime(emit, check_cancel)
    emit("download", done=0, total=0)  # barra indeterminada
    emit("status", text="Identificando hablantes…")
    with tempfile.TemporaryDirectory() as tmp:
        wav, out = Path(tmp) / "audio.wav", Path(tmp) / "hablantes.json"
        _write_wav(source, wav, check_cancel)
        cmd = [str(exe), "diarize", str(wav), "--model", str(model),
               "--preset", "v3-offline", "--format", "json", "--output", str(out), "--force"]
        try:
            _run(cmd, check_cancel)
        except RuntimeError:
            if sys.platform != "darwin":
                raise
            # Sin Metal utilizable (p. ej. una máquina virtual) se repite con la CPU.
            log.warning("nemo-speech falló con la GPU; se reintenta con la CPU", exc_info=True)
            _run(cmd + ["--device", "cpu"], check_cancel)
        segments = json.loads(out.read_text("utf-8")).get("segments", [])
    return sorted((Turn(float(s["start"]), float(s["end"]), int(s["speaker"]))
                   for s in segments), key=lambda t: t.start)


def speaker_at(turns: list[Turn], start: float, end: float) -> int | None:
    """El hablante que más tiempo habla entre `start` y `end` (None si nadie)."""
    overlap: dict[int, float] = {}
    for turn in turns:
        if turn.start >= end:
            break
        seconds = min(end, turn.end) - max(start, turn.start)
        if seconds > 0:
            overlap[turn.speaker] = overlap.get(turn.speaker, 0.0) + seconds
    return max(overlap, key=overlap.get) if overlap else None
