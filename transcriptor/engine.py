"""Motor de transcripción: descarga/carga del modelo y proceso de cada archivo.

El trabajo se ejecuta en un hilo aparte y comunica su estado mediante la
función `emit(tipo, **datos)`; la interfaz recoge esos eventos en una cola.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import audio, diarization
from .markdown import MarkdownWriter, format_timestamp
from .mlx_backend import MLX_REPOS, MlxWhisperModel, require_metal
from .paths import models_dir, output_path_for
from .power import KeepAwake

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

if sys.platform == "darwin":
    # CTranslate2 no usa la GPU de Apple: la GPU se aprovecha con MLX.
    DEVICES = [
        ("mlx", "GPU Apple Silicon (Metal, recomendado)"),
        ("cpu", "CPU Apple Silicon"),
    ]
    DEFAULT_DEVICE = "mlx"
else:
    DEVICES = [
        ("cpu", "CPU (compatible con cualquier equipo)"),
        ("cuda", "GPU NVIDIA (requiere CUDA 12 y cuDNN 9)"),
    ]
    DEFAULT_DEVICE = "cpu"
GPU_NAMES = {"cuda": "GPU NVIDIA", "mlx": "GPU del Mac"}


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
    diarize: bool = False


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


def ensure_model(name: str, emit: Emit, cancel_event: threading.Event,
                 device: str = "cpu") -> str:
    """Devuelve la ruta local del modelo, descargándolo si hace falta."""
    if device == "mlx":
        from huggingface_hub import snapshot_download as download_model
        repo = MLX_REPOS.get(name, name)
    else:
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

    device = options.device
    model = None

    def fall_back(exc: Exception) -> None:
        log.warning("No se pudo usar la GPU: %s", exc)
        emit("warning", text=(
            f"No se pudo usar la {GPU_NAMES[device]}; se usará la CPU.\n\n"
            f"Detalle: {exc}"
        ))

    if device == "mlx":
        # Antes de descargar nada: MLX necesita macOS 14 o posterior y Metal.
        try:
            require_metal()
        except Exception as exc:
            fall_back(exc)
            device = "cpu"
    if device in GPU_NAMES:
        path = ensure_model(options.model, emit, cancel_event, device)
        emit("status", text=f"Cargando el modelo «{options.model}»…")
        emit("download", done=0, total=0)  # barra indeterminada mientras carga
        try:
            if device == "mlx":
                model = MlxWhisperModel(path)
            else:
                model = WhisperModel(path, device="cuda", compute_type="float16")
        except Exception as exc:
            fall_back(exc)
            device = "cpu"
    if model is None:
        if (options.model, device) in _model_cache:
            return _model_cache[(options.model, device)], device
        path = ensure_model(options.model, emit, cancel_event)
        emit("status", text=f"Cargando el modelo «{options.model}»…")
        emit("download", done=0, total=0)  # barra indeterminada mientras carga
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
        # Sin reposo del equipo ni App Nap mientras dure el trabajo.
        with KeepAwake():
            self._run()

    def _run(self) -> None:
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
                    if device in GPU_NAMES:
                        # Errores de librerías CUDA/Metal suelen aparecer al transcribir.
                        log.exception("Error en GPU, reintentando en CPU")
                        self.emit("warning", text=(
                            f"La {GPU_NAMES[device]} falló durante la transcripción; "
                            f"se reintenta con la CPU.\n\nDetalle: {exc}"))
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

        turns = self._diarize(path) if opts.diarize else None
        speakers = None
        if turns is not None:
            count = len({t.speaker for t in turns})
            speakers = (f"{count} identificado{'s' if count != 1 else ''} con NVIDIA "
                        "Nemotron 3 Diarization (experimental)")
        output = output_path_for(path, self.out_dir)
        writer = MarkdownWriter(output, timestamps=opts.timestamps)
        language = opts.language
        started = time.monotonic()
        position = 0.0
        note = None
        # MLX entrega los segmentos al terminar cada bloque (mientras tanto solo
        # informa del avance): con bloques más cortos el texto aparece antes.
        batched = isinstance(model, MlxWhisperModel)
        chunk_seconds = 5 * 60 if batched else 20 * 60
        engine = "mlx-whisper" if batched else "faster-whisper"
        try:
            self.emit("status", text="Leyendo audio…")
            for chunk in audio.iter_audio_chunks(path, chunk_seconds=chunk_seconds,
                                                 cancel_event=self.cancel_event):
                self._check_cancel()
                self.emit("status", text="Transcribiendo…")
                extra = {}
                if batched:
                    extra["on_progress"] = (
                        lambda seconds, offset=chunk.offset:
                        self._decode_progress(offset + seconds, duration))
                segments, info = model.transcribe(
                    chunk.samples,
                    language=language,
                    task="transcribe",
                    beam_size=opts.beam_size,
                    vad_filter=opts.vad,
                    vad_parameters={"min_silence_duration_ms": 500},
                    **extra,
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
                                model=opts.model, engine=engine, speakers=speakers)
                for segment in segments:
                    self._check_cancel()
                    start = chunk.offset + segment.start
                    end = chunk.offset + segment.end
                    speaker = (diarization.speaker_at(turns, start, end)
                               if turns else None)
                    writer.add_segment(start, end, segment.text, speaker)
                    position = end
                    self.emit("segment", start=start, end=end, text=segment.text.strip(),
                              speaker=speaker)
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
        if writer.segments_written:
            _copy_srt_next_to_source(output.with_suffix(".srt"), path)
        if not writer.is_open and not output.exists():
            # Ningún bloque decodificable (archivo vacío o de menos de 0,2 s).
            writer.open(source=path, duration=duration, language=language or "—",
                        model=opts.model, engine=engine)
            writer.close(note)
        return output

    def _diarize(self, path: Path) -> list[diarization.Turn] | None:
        try:
            return diarization.diarize(path, self.emit, self._check_cancel)
        except Cancelled:
            raise
        except Exception as exc:
            log.exception("Error identificando hablantes en %s", path)
            self.emit("warning", text=(
                "No se pudo identificar a los hablantes; se transcribe sin ellos.\n\n"
                f"Detalle: {exc}"))
            return None

    def _check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled()

    def _decode_progress(self, position: float, duration: float | None) -> None:
        self._check_cancel()  # interrumpe la decodificación de MLX
        self._emit_progress(position, duration)

    def _emit_progress(self, position: float, duration: float | None) -> None:
        fraction = min(1.0, position / duration) if duration else None
        self.emit("progress", fraction=fraction, position=position, duration=duration)


def _copy_srt_next_to_source(srt: Path, source: Path) -> None:
    """Copia el .srt junto al original con su mismo nombre: los reproductores lo cargan solos.

    Un .srt que ya exista (p. ej. hecho a mano) no se pisa, y una carpeta de solo
    lectura no debe hacer fallar una transcripción ya terminada.
    """
    target = source.with_suffix(".srt")
    if target.exists():
        return
    try:
        shutil.copyfile(srt, target)
    except OSError:
        log.warning("No se pudo copiar el .srt junto a %s", source, exc_info=True)


def _friendly(exc: Exception) -> str:
    if isinstance(exc, audio.NoAudioStreamError):
        return "El archivo no contiene ninguna pista de audio."
    name = type(exc).__name__
    if name in ("InvalidDataError", "FFmpegError") or "Invalid data" in str(exc):
        return "No se pudo leer el archivo: formato no compatible o archivo dañado."
    if isinstance(exc, MemoryError):
        return "Memoria insuficiente. Prueba con un modelo más pequeño."
    return str(exc) or name
