import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from transcriptor import audio, diarization, engine
from transcriptor.markdown import MarkdownWriter, format_timestamp
from transcriptor.paths import output_path_for

SR = audio.SAMPLE_RATE


def write_wav(path: Path, samples: np.ndarray, sr: int = SR, channels: int = 1) -> Path:
    data = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())
    return path


def speech_like(seconds: float, sr: int = SR, silence_at: float | None = None) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = 0.3 * rng.standard_normal(int(seconds * sr))
    if silence_at is not None:
        a = int(silence_at * sr)
        x[a:a + sr // 2] = 0.0
    return x


def test_chunks_cover_whole_file_and_cut_in_silence(tmp_path):
    wav = write_wav(tmp_path / "a.wav", speech_like(9, silence_at=4.2))
    chunks = list(audio.iter_audio_chunks(wav, chunk_seconds=5, search_seconds=2))
    assert len(chunks) == 2
    assert sum(len(c.samples) for c in chunks) == pytest.approx(9 * SR, abs=SR // 100)
    # El corte cae dentro del tramo de silencio (4,2 s – 4,7 s).
    assert 4.2 <= chunks[1].offset <= 4.7
    assert chunks[0].samples.dtype == np.float32
    assert np.abs(chunks[0].samples).max() <= 1.0


def test_chunks_resample_stereo_44k(tmp_path):
    stereo = np.repeat(speech_like(3, sr=44100), 2)
    wav = write_wav(tmp_path / "b.wav", stereo, sr=44100, channels=2)
    chunks = list(audio.iter_audio_chunks(wav))
    assert len(chunks) == 1
    assert chunks[0].duration == pytest.approx(3, abs=0.05)
    assert audio.probe_duration(wav) == pytest.approx(3, abs=0.05)


def test_cancel_stops_decoding(tmp_path):
    wav = write_wav(tmp_path / "c.wav", speech_like(6))
    ev = threading.Event()
    ev.set()
    assert list(audio.iter_audio_chunks(wav, chunk_seconds=2, cancel_event=ev)) == []


def test_find_cut_point_prefers_silence():
    x = np.ones(SR * 4, dtype=np.float32)
    x[SR * 3: SR * 3 + 3200] = 0
    cut = audio.find_cut_point(x, SR * 4, SR * 2)
    assert SR * 3 <= cut <= SR * 3 + 3200


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(3725.9) == "01:02:05"


def test_output_path_is_unique_and_sanitized(tmp_path):
    src = Path("reunión: equipo?.mp4")
    first = output_path_for(src, tmp_path)
    assert first.name == "reunión_ equipo_ - transcripción.md"
    first.write_text("x")
    assert output_path_for(src, tmp_path).name == "reunión_ equipo_ - transcripción (2).md"
    first.unlink()
    first.with_suffix(".srt").write_text("x")  # un .srt suelto también ocupa el nombre
    assert output_path_for(src, tmp_path).name == "reunión_ equipo_ - transcripción (2).md"


def test_markdown_paragraphs(tmp_path):
    out = tmp_path / "t.md"
    w = MarkdownWriter(out, timestamps=True)
    w.open(source=Path("video.mp4"), duration=3700, language="Español", model="small")
    w.add_segment(0.0, 2.0, " Hola a todos.")
    w.add_segment(2.1, 4.0, " Empezamos.")
    w.add_segment(10.0, 12.0, " Segundo párrafo.")
    w.add_segment(12.0, 13.0, "   ")
    w.close("_fin_")
    text = out.read_text("utf-8")
    assert text.startswith("# Transcripción: video\n")
    assert "| **Duración** | 01:01:40 |" in text
    assert "**[00:00:00]** Hola a todos. Empezamos.\n\n**[00:00:10]** Segundo párrafo." in text
    assert text.rstrip().endswith("_fin_")
    assert out.with_suffix(".srt").read_text("utf-8") == (
        "1\n00:00:00,000 --> 00:00:02,000\nHola a todos.\n\n"
        "2\n00:00:02,100 --> 00:00:04,000\nEmpezamos.\n\n"
        "3\n00:00:10,000 --> 00:00:12,000\nSegundo párrafo.\n\n")
    assert out.with_suffix(".txt").read_text("utf-8") == (
        "[00:00:00] Hola a todos.\n[00:00:02] Empezamos.\n[00:00:10] Segundo párrafo.\n")


def test_markdown_speakers(tmp_path):
    out = tmp_path / "t.md"
    w = MarkdownWriter(out, timestamps=True)
    w.open(source=Path("reunion.mp4"), duration=60, language="Español", model="small",
           speakers="2 identificados")
    w.add_segment(0.0, 2.0, "Hola.", speaker=1)
    w.add_segment(2.1, 4.0, "¿Qué tal?", speaker=1)
    w.add_segment(4.1, 6.0, "Bien.", speaker=2)  # sin pausa, pero otro hablante
    w.add_segment(6.1, 7.0, "Sin asignar.")
    w.close()
    text = out.read_text("utf-8")
    assert "| **Hablantes** | 2 identificados |" in text
    assert ("**[00:00:00]** **Hablante 1:** Hola. ¿Qué tal?\n\n"
            "**[00:00:04]** **Hablante 2:** Bien.\n\n**[00:00:06]** Sin asignar.") in text
    assert "00:00:04,100 --> 00:00:06,000\nHablante 2: Bien.\n" in (
        out.with_suffix(".srt").read_text("utf-8"))
    assert "[00:00:04] Hablante 2: Bien.\n[00:00:06] Sin asignar.\n" in (
        out.with_suffix(".txt").read_text("utf-8"))


def test_speaker_at_picks_longest_overlap():
    T = diarization.Turn
    turns = [T(0, 3, 1), T(2.5, 6, 2), T(10, 12, 1)]
    assert diarization.speaker_at(turns, 0, 2) == 1
    assert diarization.speaker_at(turns, 2, 5) == 2
    assert diarization.speaker_at(turns, 6.5, 9) is None
    assert diarization.speaker_at([], 0, 1) is None


class FakeModel:
    """Sustituye a WhisperModel: un segmento por cada segundo de audio."""

    def __init__(self):
        self.calls = []

    def transcribe(self, samples, language=None, **kwargs):
        self.calls.append(language)
        n = int(len(samples) / SR)
        segs = (SimpleNamespace(start=float(i), end=i + 0.9, text=f" frase {i}")
                for i in range(n))
        return segs, SimpleNamespace(language="es", language_probability=0.97)


def run_job(tmp_path, wav, cancel_after=None, **opts):
    model = FakeModel()
    events = []
    cancel = threading.Event()

    def emit(kind, **data):
        events.append((kind, data))
        if cancel_after and kind == "segment" and len(
                [e for e in events if e[0] == "segment"]) >= cancel_after:
            cancel.set()

    engine._model_cache[("small", "cpu")] = model
    try:
        job = engine.TranscriptionJob([wav], engine.Options(**opts), tmp_path, emit, cancel)
        job.run()
    finally:
        engine._model_cache.clear()
    return model, events


def test_job_writes_markdown_with_offsets(tmp_path, monkeypatch):
    monkeypatch.setattr(audio, "iter_audio_chunks",
                        lambda p, cancel_event=None, _f=audio.iter_audio_chunks, **_kw:
                        _f(p, chunk_seconds=4, search_seconds=1, cancel_event=cancel_event))
    wav = write_wav(tmp_path / "entrevista.wav", speech_like(9))
    model, events = run_job(tmp_path, wav, timestamps=False)
    kinds = [k for k, _ in events]
    assert kinds[-1] == "finished" and "file_done" in kinds
    # El idioma detectado en el primer bloque se fija para los siguientes.
    assert model.calls[0] is None and all(c == "es" for c in model.calls[1:])
    progress = [d["fraction"] for k, d in events if k == "progress" and d["fraction"]]
    assert progress == sorted(progress) and progress[-1] == pytest.approx(1.0, abs=0.01)
    starts = [d["start"] for k, d in events if k == "segment"]
    assert starts == sorted(starts) and starts[-1] > 4  # desplazamiento de bloques aplicado
    out = next(d["output"] for k, d in events if k == "file_done")
    text = out.read_text("utf-8")
    assert "Español (detectado, 97%)" in text
    assert "frase 0" in text and "completada" in text


def test_job_cancel_keeps_partial_text(tmp_path):
    wav = write_wav(tmp_path / "larga.wav", speech_like(8))
    _model, events = run_job(tmp_path, wav, cancel_after=3)
    kinds = [k for k, _ in events]
    assert "cancelled" in kinds and "file_done" not in kinds
    out = tmp_path / "larga - transcripción.md"
    text = out.read_text("utf-8")
    assert "frase 2" in text and "cancelada" in text and "frase 5" not in text


def source_in_subdir(tmp_path, name="entrevista.wav"):
    src = tmp_path / "origen"
    src.mkdir()
    return write_wav(src / name, speech_like(8))


def test_job_copies_srt_next_to_source(tmp_path):
    wav = source_in_subdir(tmp_path)
    _model, events = run_job(tmp_path, wav)
    out = next(d["output"] for k, d in events if k == "file_done")
    sidecar = wav.with_suffix(".srt")
    assert sidecar.read_text("utf-8") == out.with_suffix(".srt").read_text("utf-8")
    assert "frase 0" in sidecar.read_text("utf-8")


def test_job_keeps_existing_srt_next_to_source(tmp_path):
    wav = source_in_subdir(tmp_path)
    wav.with_suffix(".srt").write_text("hecho a mano", "utf-8")
    _model, events = run_job(tmp_path, wav)
    assert "file_done" in [k for k, _ in events]
    assert wav.with_suffix(".srt").read_text("utf-8") == "hecho a mano"


def test_job_cancel_leaves_no_srt_next_to_source(tmp_path):
    wav = source_in_subdir(tmp_path)
    run_job(tmp_path, wav, cancel_after=3)
    assert not wav.with_suffix(".srt").exists()


def test_job_survives_unwritable_source_folder(tmp_path, monkeypatch):
    wav = source_in_subdir(tmp_path)

    def denied(*_a, **_kw):
        raise PermissionError("solo lectura")

    monkeypatch.setattr(engine.shutil, "copyfile", denied)
    _model, events = run_job(tmp_path, wav)
    kinds = [k for k, _ in events]
    assert "file_done" in kinds and "file_error" not in kinds
    assert not wav.with_suffix(".srt").exists()


def test_job_reports_file_without_audio(tmp_path):
    bad = tmp_path / "roto.mp4"
    bad.write_bytes(b"esto no es un video")
    _model, events = run_job(tmp_path, bad)
    errors = [d for k, d in events if k == "file_error"]
    assert errors and "roto.mp4" in str(errors[0]["path"])
    assert events[-1][0] == "finished"


def test_power_helpers():
    import sys
    from transcriptor.power import KeepAwake, disable_background_throttling
    # En Windows la llamada debe aceptarse; en el resto no hace nada.
    assert disable_background_throttling() is (sys.platform == "win32")
    with KeepAwake():
        pass


def test_job_labels_speakers(tmp_path, monkeypatch):
    T = diarization.Turn
    monkeypatch.setattr(diarization, "diarize",
                        lambda path, emit, check_cancel: [T(0, 2.95, 1), T(3, 9, 2)])
    wav = write_wav(tmp_path / "reunion.wav", speech_like(5))
    _model, events = run_job(tmp_path, wav, diarize=True)
    speakers = [d["speaker"] for k, d in events if k == "segment"]
    assert speakers == [1, 1, 1, 2, 2]
    out = next(d["output"] for k, d in events if k == "file_done")
    text = out.read_text("utf-8")
    assert "2 identificados" in text and "**Hablante 2:** frase 3" in text


def test_job_continues_when_diarization_fails(tmp_path, monkeypatch):
    def broken(path, emit, check_cancel):
        raise RuntimeError("sin conexión")

    monkeypatch.setattr(diarization, "diarize", broken)
    wav = write_wav(tmp_path / "reunion.wav", speech_like(3))
    _model, events = run_job(tmp_path, wav, diarize=True)
    kinds = [k for k, _ in events]
    assert "file_done" in kinds and "file_error" not in kinds
    assert any("sin conexión" in d["text"] for k, d in events if k == "warning")
    out = next(d["output"] for k, d in events if k == "file_done")
    assert "Hablante" not in out.read_text("utf-8")
