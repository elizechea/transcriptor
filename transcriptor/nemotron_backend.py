"""Transcripción con NVIDIA Nemotron 3.5 Transcribe (30B-A4B) en la GPU del Mac.

Es un modelo de lenguaje con oído: más preciso que Whisper, pero pesa 22 GB
(versión NVFP4, la única que trae las marcas de tiempo por palabra) y necesita
un Mac con 32 GB de memoria o más. Corre con MLX mediante mlx-audio, así que
solo existe en macOS. No transcribe gallego, catalán ni euskera.

Expone la misma interfaz que `faster_whisper.WhisperModel.transcribe`.
"""

from __future__ import annotations

import os
import sys
import types
from itertools import permutations
from typing import Callable

import numpy as np

from .cactus_backend import _sentences
from .mlx_backend import Info

NEMOTRON_REPO = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
NEMOTRON_UNSUPPORTED = ("gl", "ca", "eu")
MIN_MEMORY = 32 * 1024**3


def check_memory() -> None:
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    if total < MIN_MEMORY:
        raise RuntimeError(
            f"El modelo Nemotron necesita un Mac con 32 GB de memoria y este tiene "
            f"{total / 1024**3:.0f} GB. Elige otro modelo.")


def linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Asignación de coste mínimo, como la de scipy.optimize, por fuerza bruta.

    mlx-audio la usa para casar las etiquetas de hablante del texto (como mucho 8)
    con los hablantes del audio; scipy no va en la app (ver Transcriptor.spec).
    """
    cost = np.asarray(cost)
    if cost.shape[0] > cost.shape[1]:
        cols, rows = linear_sum_assignment(cost.T)
        order = np.argsort(rows)
        return rows[order], cols[order]
    rows = np.arange(cost.shape[0])
    best = min(permutations(range(cost.shape[1]), len(rows)),
               key=lambda cols: cost[rows, list(cols)].sum())
    return rows, np.array(best, dtype=int)


def _without_scipy() -> None:
    try:
        import scipy.optimize  # noqa: F401
    except ImportError:
        stub = types.ModuleType("scipy.optimize")
        stub.linear_sum_assignment = linear_sum_assignment
        # Sin módulo «scipy» falso: transformers lo detectaría como instalado.
        sys.modules["scipy.optimize"] = stub


class NemotronModel:
    def __init__(self, path: str):
        _without_scipy()
        from mlx_audio.stt import load

        self._model = load(path, strict=True)

    def transcribe(self, samples: np.ndarray, language: str | None = None,
                   on_progress: Callable[[float], None] | None = None, **_options):
        """Transcribe `samples` (float32 mono 16 kHz). VAD y beam_size se ignoran."""
        result = None
        for result in self._model.generate(samples, language=language, stream=True,
                                           word_timestamps=True, speculative=True):
            if on_progress:
                on_progress(0.0)  # el texto llega sin tiempos: solo sirve para cancelar
        words = [{"word": w["text"], "start": w["start"], "end": w["end"]}
                 for w in (result.segments or [])] if result else []
        # Nemotron no dice qué idioma ha detectado.
        return iter(_sentences(words)), Info(language, None)
