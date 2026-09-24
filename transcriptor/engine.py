"""Motor de transcripción: descarga/carga del modelo y proceso de cada archivo.

El trabajo se ejecuta en un hilo aparte y comunica su estado mediante la
función `emit(tipo, **datos)`; la interfaz recoge esos eventos en una cola.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import audio
from .markdown import MarkdownWriter, format_timestamp
from .paths import models_dir, output_path_for

log = logging.getLogger(__name__)

# (clave, etiqueta visible, tamaño aproximado de descarga en bytes)
MODELS = [
    ("tiny", "Tiny — muy rápido, precisión baja", 75_000_000),
    ("base", "Base — rápido, precisión aceptable", 145_000_000),
    ("small", "Small — equilibrado (recomendado)", 484_000_000),
    ("medium", "Medium — preciso, lento", 1_530_000_000),
    ("large-v3-turbo", "Large v3 Turbo — muy preciso, velocidad media", 1_620_000_000),
    ("large-v3", "Large v3 — máxima precisión, muy lento en CPU", 3_090_000_000),
]
MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "large-v3": "Systran/faster-whisper-large-v3",
}

LANGUAGES = [
    (None, "Detectar automáticamente"),
    ("es", "Español"),
    ("en", "Inglés"),
    ("gl", "Gallego"),
    ("ca", "Catalán"),
    ("eu", "Euskera"),
    ("pt", "Portugués"),
    ("fr", "Francés"),
    ("it", "Italiano"),
    ("de", "Alemán"),
    ("nl", "Neerlandés"),
    ("ru", "Ruso"),
    ("zh", "Chino"),
    ("ja", "Japonés"),
    ("ar", "Árabe"),
]
LANGUAGE_NAMES = {code: name for code, name in LANGUAGES if code}

DEVICES = [
    ("cpu", "CPU (compatible con cualquier equipo)"),
    ("cuda", "GPU NVIDIA (requiere CUDA 12 y cuDNN 9)"),
]


class Cancelled(Exception):
    pass


@dataclass
class Options:
    model: str = "small"
    language: str | None = None
    device: str = "cpu"
    timestamps: bool = True
    vad: bool = True
    beam_size: int = 5


Emit = Callable[..., None]

_model_cache: dict[tuple[str, str], object] = {}


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def ensure_model(name: str, emit: Emit, cancel_event: threading.Event) -> str:
    """Devuelve la ruta local del modelo, descargándolo si hace falta."""
    from faster_whisper.utils import download_model

    repo = MODEL_REPOS.get(name, name)
    cache = models_dir()
    try:
        return download_model(repo, cache_dir=str(cache), local_files_only=True)
    except Exception:
        pass

    expected = next((size for key, _label, size in MODELS if key == name), 0)
    repo_dir = cache / ("models--" + repo.replace("/", "--"))
    emit("status", text=f"Descargando el modelo «{name}» (solo la primera vez)…")
    stop = threading.Event()

    def watch():
        start = _dir_size(repo_dir) if repo_dir.exists() else 0
        while not stop.wait(0.5):
            done = (_dir_size(repo_dir) if repo_dir.exists() else 0) - start
            emit("download", done=max(0, done), total=expected)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        path = download_model(repo, cache_dir=str(cache))
    except Exception as exc:
        raise RuntimeError(
            "No se pudo descargar el modelo. Comprueba la conexión a Internet "
            f"(se descarga desde huggingface.co).\n\nDetalle: {exc}"
        ) from exc
    finally:
        stop.set()
        watcher.join(timeout=2)
    if cancel_event.is_set():
        raise Cancelled()
    return path


def load_model(options: Options, emit: Emit, cancel_event: threading.Event):
    from faster_whisper import WhisperModel

    key = (options.model, options.device)
    if key in _model_cache:
        return _model_cache[key], options.device

    path = ensure_model(options.model, emit, cancel_event)
    emit("status", text=f"Cargando el modelo «{options.model}»…")
    emit("download", done=0, total=0)  # barra indeterminada mientras carga
    device = options.device
    model = None
    if device == "cuda":
        try:
            model = WhisperModel(path, device="cuda", compute_type="float16")
        except Exception as exc:
            log.warning("No se pudo usar la GPU: %s", exc)
            emit("warning", text=(
                "No se pudo usar la GPU NVIDIA; se usará la CPU.\n\n"
                f"Detalle: {exc}"
            ))
            device = "cpu"
    if model is None:
        model = WhisperModel(
            path, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4
        )
    _model_cache[(options.model, device)] = model
    return model, device


class TranscriptionJob:
    def __init__(self, files: list[Path], options: Options, out_dir: Path,
                 emit: Emit, cancel_event: threading.Event):
        self.files = files
        self.options = options
        self.out_dir = out_dir
        self.emit = emit
        self.cancel_event = cancel_event

    def run(self) -> None:
        try:
            model, device = load_model(self.options, self.emit, self.cancel_event)
            for index, path in enumerate(self.files):
                if self.cancel_event.is_set():
                    raise Cancelled()
                self.emit("file_start", index=index, total=len(self.files), path=path)
                try:
                    output = self._transcribe(path, model)
                except Cancelled:
                    raise
                except Exception as exc:
                    if device == "cuda":
                        # Errores de librerías CUDA suelen aparecer al transcribir.
                        log.exception("Error en GPU, reintentando en CPU")
                        self.emit("warning", text=(
                            "La GPU falló durante la transcripción; se reintenta "
                            f"con la CPU.\n\nDetalle: {exc}"))
                        self.options.device = device = "cpu"
                        model, device = load_model(self.options, self.emit,
                                                   self.cancel_event)
                        self.emit("file_start", index=index, total=len(self.files),
                                  path=path)
                        try:
                            output = self._transcribe(path, model)
                        except Cancelled:
                            raise
                        except Exception as exc2:
                            log.exception("Error transcribiendo %s", path)
                            self.emit("file_error", path=path, message=str(exc2))
                            continue
                    else:
                        log.exception("Error transcribiendo %s", path)
                        self.emit("file_error", path=path, message=_friendly(exc))
                        continue
                self.emit("file_done", path=path, output=output)
        except Cancelled:
            self.emit("cancelled")
        except Exception as exc:
            log.exception("Error general")
            self.emit("fatal", message=_friendly(exc))
        finally:
            self.emit("finished")

    def _transcribe(self, path: Path, model) -> Path:
        opts = self.options
        self.emit("status", text="Analizando el archivo…")
        duration = audio.probe_duration(path)
        self.emit("progress", fraction=0.0 if duration else None, position=0.0,
                  duration=duration)

        output = output_path_for(path, self.out_dir)
        writer = MarkdownWriter(output, timestamps=opts.timestamps)
        language = opts.language
        started = time.monotonic()
        position = 0.0
        note = None
        try:
            self.emit("status", text="Leyendo audio…")
            for chunk in audio.iter_audio_chunks(path, cancel_event=self.cancel_event):
                self._check_cancel()
                self.emit("status", text="Transcribiendo…")
                segments, info = model.transcribe(
                    chunk.samples,
                    language=language,
                    task="transcribe",
                    beam_size=opts.beam_size,
                    vad_filter=opts.vad,
                    vad_parameters={"min_silence_duration_ms": 500},
                )
                if not writer.is_open:
                    if language is None:
                        language = info.language
                        label = (f"{LANGUAGE_NAMES.get(language, language)} "
                                 f"(detectado, {info.language_probability:.0%})")
                        self.emit("language", code=language,
                                  probability=info.language_probability)
                    else:
                        label = LANGUAGE_NAMES.get(language, language)
                    writer.open(source=path, duration=duration, language=label,
                                model=opts.model)
                for segment in segments:
                    self._check_cancel()
                    start = chunk.offset + segment.start
                    end = chunk.offset + segment.end
                    writer.add_segment(start, end, segment.text)
                    position = end
                    self.emit("segment", start=start, end=end, text=segment.text.strip())
                    self._emit_progress(position, duration)
                position = chunk.offset + chunk.duration
                self._emit_progress(position, duration)
                self.emit("status", text="Leyendo audio…")
            self._check_cancel()  # el decodificador se detiene sin excepción
            elapsed = format_timestamp(time.monotonic() - started)
            note = f"_Transcripción completada en {elapsed} con Transcriptor._"
        except Cancelled:
            note = (f"_⚠️ Transcripción cancelada por el usuario en "
                    f"{format_timestamp(position)}; el texto está incompleto._")
            raise
        except Exception as exc:
            note = (f"_⚠️ La transcripción se interrumpió por un error en "
                    f"{format_timestamp(position)}: {exc}_")
            raise
        finally:
            writer.close(note)  # no hace nada si el archivo no llegó a abrirse
        if not writer.is_open and not output.exists():
            # Ningún bloque decodificable (archivo vacío o de menos de 0,2 s).
            writer.open(source=path, duration=duration, language=language or "—",
                        model=opts.model)
            writer.close(note)
        return output

    def _check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled()

    def _emit_progress(self, position: float, duration: float | None) -> None:
        fraction = min(1.0, position / duration) if duration else None
        self.emit("progress", fraction=fraction, position=position, duration=duration)


def _friendly(exc: Exception) -> str:
    if isinstance(exc, audio.NoAudioStreamError):
        return "El archivo no contiene ninguna pista de audio."
    name = type(exc).__name__
    if name in ("InvalidDataError", "FFmpegError") or "Invalid data" in str(exc):
        return "No se pudo leer el archivo: formato no compatible o archivo dañado."
    if isinstance(exc, MemoryError):
        return "Memoria insuficiente. Prueba con un modelo más pequeño."
    return str(exc) or name
