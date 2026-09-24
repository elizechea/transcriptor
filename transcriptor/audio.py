"""Decodificación de audio en streaming para archivos de gran tamaño.

faster-whisper, por defecto, decodifica el archivo completo en memoria
(unos 230 MB por hora de audio). Para archivos de varias horas eso resulta
costoso, así que aquí se decodifica con PyAV poco a poco y se entregan
bloques de ~20 minutos, cortados en el punto más silencioso cercano al
límite para no partir palabras.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

SAMPLE_RATE = 16000

AUDIO_EXTENSIONS = (
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma",
    ".aiff", ".aif", ".amr", ".mka",
)
VIDEO_EXTENSIONS = (
    ".mp4", ".mkv", ".mov", ".avi", ".wmv", ".webm", ".flv", ".m4v", ".mpg",
    ".mpeg", ".ts", ".mts", ".m2ts", ".3gp", ".vob", ".ogv",
)
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS + VIDEO_EXTENSIONS


class NoAudioStreamError(Exception):
    """El archivo no contiene ninguna pista de audio."""


@dataclass
class AudioChunk:
    offset: float  # segundos desde el inicio del archivo
    samples: np.ndarray  # float32 mono a 16 kHz

    @property
    def duration(self) -> float:
        return len(self.samples) / SAMPLE_RATE


def probe_duration(path: Path) -> float | None:
    """Duración del archivo en segundos, o None si el contenedor no la indica."""
    import av

    with av.open(str(path), metadata_errors="ignore") as container:
        if not container.streams.audio:
            raise NoAudioStreamError(path.name)
        stream = container.streams.audio[0]
        if stream.duration and stream.time_base:
            return float(stream.duration * stream.time_base)
        if container.duration:
            return container.duration / 1_000_000  # av.time_base
    return None


def find_cut_point(samples: np.ndarray, target: int, search: int, window: int = 1600) -> int:
    """Índice cercano a `target` (hacia atrás, hasta `search` muestras) con menor energía."""
    start = max(0, target - search)
    region = samples[start:target]
    n_windows = len(region) // window
    if n_windows == 0:
        return target
    frames = region[: n_windows * window].astype(np.float32).reshape(n_windows, window)
    energy = np.mean(frames * frames, axis=1)
    best = int(np.argmin(energy))
    return start + best * window + window // 2


def iter_audio_chunks(
    path: Path,
    chunk_seconds: float = 20 * 60,
    search_seconds: float = 30,
    cancel_event=None,
) -> Iterator[AudioChunk]:
    """Decodifica `path` y devuelve bloques de audio mono a 16 kHz."""
    import av

    chunk_len = int(chunk_seconds * SAMPLE_RATE)
    search_len = int(search_seconds * SAMPLE_RATE)
    pending: list[np.ndarray] = []
    pending_len = 0
    offset_samples = 0

    def take(cut: int) -> AudioChunk:
        nonlocal pending, pending_len, offset_samples
        data = np.concatenate(pending) if len(pending) > 1 else pending[0]
        head, tail = data[:cut], data[cut:]
        pending = [tail] if len(tail) else []
        pending_len = len(tail)
        chunk = AudioChunk(offset_samples / SAMPLE_RATE, head.astype(np.float32) / 32768.0)
        offset_samples += cut
        return chunk

    with av.open(str(path), metadata_errors="ignore") as container:
        if not container.streams.audio:
            raise NoAudioStreamError(path.name)
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)

        def frames():
            for packet in container.demux(stream):
                if cancel_event is not None and cancel_event.is_set():
                    return
                try:
                    decoded = packet.decode()
                except av.error.InvalidDataError:
                    continue  # paquete corrupto: se ignora, como hace faster-whisper
                for frame in decoded:
                    yield from resampler.resample(frame)
            yield from resampler.resample(None)  # vacía el resampler

        for frame in frames():
            array = frame.to_ndarray().reshape(-1)
            if not len(array):
                continue
            pending.append(array)
            pending_len += len(array)
            if pending_len >= chunk_len:
                data = np.concatenate(pending)
                pending = [data]
                yield take(find_cut_point(data, chunk_len, search_len))

    if cancel_event is not None and cancel_event.is_set():
        return
    if pending_len >= SAMPLE_RATE // 5:  # descarta restos de menos de 0,2 s
        yield take(pending_len)
