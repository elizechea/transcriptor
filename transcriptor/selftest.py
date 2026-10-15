"""Autocomprobación del ejecutable (se usa en CI para validar el .exe).

    Transcriptor.exe --self-test [--model tiny] [--report ruta.txt]

Verifica que todas las dependencias nativas (PyAV, CTranslate2, onnxruntime)
están empaquetadas y funcionan. Con --model también descarga el modelo y
transcribe un audio sintético de principio a fin.
"""

from __future__ import annotations

import sys
import tempfile
import threading
import traceback
import wave
from pathlib import Path

import numpy as np


def _arg(argv: list[str], name: str) -> str | None:
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


def _make_wav(path: Path, seconds: float = 4.0) -> None:
    sr = 16000
    t = np.arange(int(sr * seconds)) / sr
    tone = 0.3 * np.sin(2 * np.pi * 440 * t) * (t % 1 < 0.5)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((tone * 32767).astype("<i2").tobytes())


def run_self_test(argv: list[str]) -> int:
    report_path = _arg(argv, "--report")
    model = _arg(argv, "--model")
    lines: list[str] = []
    ok = True

    def check(name, fn):
        nonlocal ok
        try:
            result = fn()
            lines.append(f"OK   {name}: {result}")
        except Exception:
            ok = False
            lines.append(f"FAIL {name}:\n{traceback.format_exc()}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        wav = tmp_path / "test.wav"
        _make_wav(wav)

        def decode():
            from transcriptor.audio import iter_audio_chunks, probe_duration
            chunks = list(iter_audio_chunks(wav, chunk_seconds=1.5, search_seconds=0.5))
            total = sum(c.duration for c in chunks)
            return f"{len(chunks)} bloques, {total:.2f}s (duración {probe_duration(wav):.2f}s)"

        def ctranslate():
            import ctranslate2
            return f"ctranslate2 {ctranslate2.__version__}, CUDA: {ctranslate2.get_cuda_device_count()}"

        def mlx():
            from transcriptor.mlx_backend import require_metal
            try:
                return f"GPU: {require_metal()}"
            except RuntimeError as exc:
                # Sin GPU (p. ej. máquina virtual) la app usa la CPU: no es un fallo.
                if "Metal" not in str(exc):
                    raise
                return f"sin GPU, se usará la CPU ({exc})"

        def vad():
            from faster_whisper.vad import get_speech_timestamps
            from transcriptor.audio import iter_audio_chunks
            audio = next(iter(iter_audio_chunks(wav))).samples
            return f"{len(get_speech_timestamps(audio))} tramos"

        def tk():
            import tkinter
            return f"Tk {tkinter.TkVersion}"

        def downloads():
            from transcriptor.paths import downloads_dir
            return str(downloads_dir())

        check("decodificación PyAV", decode)
        check("CTranslate2", ctranslate)
        if sys.platform == "darwin":
            check("MLX (GPU Apple Silicon)", mlx)
        check("VAD onnxruntime", vad)
        check("tkinter", tk)
        check("carpeta Descargas", downloads)

        if sys.platform == "win32":
            def throttling():
                from transcriptor.power import disable_background_throttling
                if not disable_background_throttling():
                    raise RuntimeError("SetProcessInformation(ProcessPowerThrottling) falló")
                return "EcoQoS desactivado para el proceso"

            check("sin limitación en segundo plano", throttling)

        if model:
            def transcribe():
                from transcriptor.engine import DEFAULT_DEVICE, Options, TranscriptionJob
                events = []
                job = TranscriptionJob(
                    [wav], Options(model=model, device=DEFAULT_DEVICE, vad=False), tmp_path,
                    lambda kind, **data: events.append((kind, data)), threading.Event())
                job.run()
                kinds = [k for k, _ in events]
                fails = [d for k, d in events if k in ("fatal", "file_error")]
                if fails or "file_done" not in kinds:
                    raise RuntimeError(f"eventos: {kinds} errores: {fails}")
                out = next(d["output"] for k, d in events if k == "file_done")
                warnings = [d["text"].splitlines()[0] for k, d in events if k == "warning"]
                return f"{out.name} ({out.stat().st_size} bytes) {' '.join(warnings)}".strip()

            check(f"transcripción con modelo {model}", transcribe)

    text = "\n".join(lines) + f"\n\nRESULTADO: {'OK' if ok else 'FALLO'}\n"
    print(text)
    if report_path:
        Path(report_path).write_text(text, encoding="utf-8")
    return 0 if ok else 1
