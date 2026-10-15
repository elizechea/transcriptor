"""Transcripción en la GPU de los Mac con chip M mediante MLX (mlx-whisper).

CTranslate2, el motor de faster-whisper, no tiene soporte para Metal, así que
en macOS solo puede usar la CPU. Este módulo envuelve mlx-whisper con la misma
interfaz que `faster_whisper.WhisperModel.transcribe` (segmentos con
`start`/`end`/`text` e `info.language`/`info.language_probability`) para que
el motor no tenga que distinguir entre ambos.

El filtro VAD es el mismo de faster-whisper (Silero), aplicado antes de
pasar el audio a MLX; las marcas de tiempo se devuelven al tiempo original.
"""

from __future__ import annotations

import importlib
import sys
import threading
import types
from dataclasses import dataclass
from typing import Callable

import numpy as np

SAMPLE_RATE = 16000

# Mismos modelos que en faster-whisper, convertidos a MLX (float16).
MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}

# mlx-whisper guarda un único modelo en una variable global y su bucle usa una
# barra tqdm global: no admite dos transcripciones a la vez.
_lock = threading.Lock()


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Info:
    language: str
    language_probability: float


def require_metal() -> str:
    """Comprueba que MLX se puede usar y devuelve el nombre de la GPU."""
    try:
        import mlx.core as mx
    except (ImportError, OSError) as exc:  # p. ej. macOS 13: MLX pide macOS 14
        raise RuntimeError(f"MLX no está disponible en este sistema ({exc})") from exc
    if not mx.metal.is_available():
        raise RuntimeError("Metal no está disponible en este equipo")
    return mx.device_info().get("device_name", "GPU")


def _skip_word_timestamps() -> None:
    """mlx_whisper.timing importa numba y scipy (~150 MB en la app) solo para
    las marcas de tiempo por palabra, que no se usan: si no están instalados
    (la app no los incluye), se sustituye por un módulo vacío."""
    try:
        import numba  # noqa: F401
        import scipy.signal  # noqa: F401
    except ImportError:
        def add_word_timestamps(*_args, **_kwargs):
            raise NotImplementedError("marcas de tiempo por palabra no disponibles")

        stub = types.ModuleType("mlx_whisper.timing")
        stub.add_word_timestamps = add_word_timestamps
        sys.modules.setdefault("mlx_whisper.timing", stub)


class MlxWhisperModel:
    def __init__(self, path: str):
        require_metal()
        _skip_word_timestamps()

        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        self.path = path
        self._model = ModelHolder.get_model(path, mx.float16)

    def transcribe(self, samples: np.ndarray, language: str | None = None,
                   task: str = "transcribe", beam_size: int | None = None,
                   vad_filter: bool = False, vad_parameters: dict | None = None,
                   on_progress: Callable[[float], None] | None = None):
        """Transcribe `samples` (float32 mono 16 kHz).

        `beam_size` se ignora: mlx-whisper solo implementa búsqueda voraz.
        `on_progress(segundos)` se llama a medida que avanza la decodificación
        (en tiempo del audio original); si lanza una excepción, se interrumpe.
        """
        import mlx_whisper

        speech_map = None
        audio = samples
        if vad_filter:
            from faster_whisper.vad import (SpeechTimestampsMap, VadOptions,
                                            collect_chunks, get_speech_timestamps)

            speech = get_speech_timestamps(samples, VadOptions(**(vad_parameters or {})))
            if not speech:
                return iter(()), Info(language or "", 1.0 if language else 0.0)
            pieces, _meta = collect_chunks(samples, speech)
            audio = np.concatenate(pieces).astype(np.float32)
            speech_map = SpeechTimestampsMap(speech, SAMPLE_RATE)

        def original(t: float, is_end: bool = False) -> float:
            return speech_map.get_original_time(t, is_end=is_end) if speech_map else t

        with _lock:
            probability = 1.0
            if language is None:
                language, probability = self._detect_language(audio)

            module = importlib.import_module("mlx_whisper.transcribe")
            saved = module.tqdm
            module.tqdm = _ProgressTqdm.module(
                lambda frames: on_progress(original(frames / 100.0))
                if on_progress else None)
            try:
                result = mlx_whisper.transcribe(
                    audio,
                    path_or_hf_repo=self.path,
                    language=language,
                    task=task,
                    verbose=False,  # sin texto en stdout; activa la barra (nuestro hook)
                    fp16=True,
                )
            finally:
                module.tqdm = saved

        segments = [
            Segment(original(s["start"]), original(s["end"], is_end=True), s["text"])
            for s in result["segments"]
        ]
        return iter(segments), Info(language, probability)

    def _detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

        if not self._model.is_multilingual:
            return "en", 1.0
        mel = log_mel_spectrogram(audio[:N_SAMPLES], n_mels=self._model.dims.n_mels,
                                  padding=N_SAMPLES)
        mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        _tokens, probs = self._model.detect_language(mel)
        code = max(probs, key=probs.get)
        return code, float(probs[code])


class _ProgressTqdm:
    """Sustituto de `tqdm.tqdm` que informa del avance (en frames de 10 ms)."""

    def __init__(self, callback, total=None, **_kwargs):
        self.callback = callback
        self.n = 0

    @classmethod
    def module(cls, callback):
        from types import SimpleNamespace
        return SimpleNamespace(tqdm=lambda *a, **kw: cls(callback, *a, **kw))

    def update(self, n=1):
        self.n += n
        self.callback(self.n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
