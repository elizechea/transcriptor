"""Transcripción con Cactus Whistle, un modelo de voz de 17 MB que corre en la CPU.

Solo transcribe siete idiomas (WHISTLE_LANGUAGES), pero va cientos de veces más
rápido que el tiempo real y no necesita filtro VAD: el silencio y el ruido no le
hacen inventar texto. Ni el motor nativo (libcactus, ~1,5 MB) ni el modelo van
dentro de la app: se descargan la primera vez, con las huellas SHA-256 que fija
cactus-compute, y se guardan junto a los demás modelos.

Expone la misma interfaz que `faster_whisper.WhisperModel.transcribe`.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

from .mlx_backend import Info, Segment
from .paths import models_dir

WHISTLE_LANGUAGES = ("es", "en", "fr", "it", "de", "nl", "pl")
# Silencio entre dos palabras (segundos) que corta la frase aunque no haya punto.
_PAUSE = 1.0


def ensure_whistle(emit: Callable[..., None]) -> tuple[str, str]:
    """Rutas del motor libcactus y del modelo, descargándolos si hace falta."""
    from cactus.agent import fetch

    lib = models_dir() / f"cactus-{fetch.ENGINE_VERSION}" / fetch.lib_name()
    folder = models_dir() / f"whistle-{fetch.MODEL_REVISIONS[fetch.WHISTLE][:12]}"
    weights = folder / fetch.base_weights(fetch.WHISTLE)
    if not (lib.exists() and weights.exists()):
        emit("status", text="Descargando el modelo «whistle» (solo la primera vez)…")
        emit("download", done=0, total=0)  # ~18 MB: basta la barra indeterminada
        try:
            fetch.fetch_library(dest_dir=str(lib.parent))
            fetch.fetch_weights(fetch.WHISTLE, dest_dir=str(folder))
        except Exception as exc:
            raise RuntimeError(
                "No se pudo descargar el modelo. Comprueba la conexión a Internet "
                f"(se descarga desde huggingface.co).\n\nDetalle: {exc}"
            ) from exc
    return str(lib), str(weights)


class WhistleModel:
    def __init__(self, lib: str, weights: str):
        # Con la ruta fijada, cactus no consulta Hugging Face cada vez que carga el motor.
        os.environ["CACTUS_LIB_PATH"] = lib
        os.environ.setdefault("CACTUS_TELEMETRY", "0")
        import cactus

        self._model = cactus.Whistle(weights=weights)

    def transcribe(self, samples: np.ndarray, language: str | None = None, **_options):
        """Transcribe `samples` (float32 mono 16 kHz). VAD y beam_size se ignoran."""
        result = self._model.transcribe(samples, language=language, word_timestamps=True)
        # Whistle no da la probabilidad del idioma detectado.
        return iter(_sentences(result.get("words") or [])), Info(result["language"], None)


def _sentences(words: list[dict]) -> list[Segment]:
    """Agrupa las palabras en frases: corta en . ? ! … o en una pausa larga."""
    segments, current = [], []
    for i, word in enumerate(words):
        current.append(word)
        if (i + 1 == len(words) or word["word"].endswith((".", "?", "!", "…"))
                or words[i + 1]["start"] - word["end"] >= _PAUSE):
            text = " " + " ".join(w["word"] for w in current)
            segments.append(Segment(current[0]["start"], current[-1]["end"], text))
            current = []
    return segments
