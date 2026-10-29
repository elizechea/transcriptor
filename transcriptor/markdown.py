"""Escritura incremental de la transcripción en Markdown, SRT y TXT.

Junto al .md se escriben, con el mismo nombre, unos subtítulos .srt y un .txt
con una línea por segmento y su marca de tiempo (como la vista previa).
Si se identificaron los hablantes, cada cambio de hablante abre un párrafo.

Los párrafos se escriben en disco en cuanto se cierran, de modo que en
archivos muy largos el resultado parcial no se pierde si algo falla.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

PARAGRAPH_GAP = 2.0  # segundos de silencio que abren un párrafo nuevo
PARAGRAPH_MAX_CHARS = 700


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_srt_timestamp(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def speaker_label(speaker: int) -> str:
    return f"Hablante {speaker}"


def _md_escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


class MarkdownWriter:
    def __init__(self, path: Path, timestamps: bool = True):
        self.path = path
        self.timestamps = timestamps
        self._file = None
        self._srt = None
        self._txt = None
        self._para_text: list[str] = []
        self._para_start = 0.0
        self._para_speaker: int | None = None
        self._last_end: float | None = None
        self.segments_written = 0

    @property
    def is_open(self) -> bool:
        return self._file is not None

    def open(self, *, source: Path, duration: float | None, language: str,
             model: str, engine: str = "faster-whisper",
             speakers: str | None = None) -> None:
        self._file = open(self.path, "w", encoding="utf-8", newline="\n")
        self._srt = open(self.path.with_suffix(".srt"), "w", encoding="utf-8", newline="\n")
        self._txt = open(self.path.with_suffix(".txt"), "w", encoding="utf-8", newline="\n")
        rows = [
            ("Archivo", f"`{_md_escape_cell(source.name)}`"),
            ("Fecha", datetime.now().strftime("%d/%m/%Y %H:%M")),
            ("Duración", format_timestamp(duration) if duration else "desconocida"),
            ("Idioma", language),
            ("Modelo", f"{engine} `{model}`"),
        ]
        if speakers:
            rows.append(("Hablantes", speakers))
        self._file.write(f"# Transcripción: {source.stem}\n\n")
        self._file.write("| | |\n|---|---|\n")
        for key, value in rows:
            self._file.write(f"| **{key}** | {value} |\n")
        self._file.write("\n---\n\n")
        self._file.flush()

    def add_segment(self, start: float, end: float, text: str,
                    speaker: int | None = None) -> None:
        text = text.strip()
        if not text:
            return
        if self._para_text and (
            (self._last_end is not None and start - self._last_end >= PARAGRAPH_GAP)
            or sum(len(t) + 1 for t in self._para_text) >= PARAGRAPH_MAX_CHARS
            or speaker != self._para_speaker
        ):
            self._flush_paragraph()
        if not self._para_text:
            self._para_start = start
            self._para_speaker = speaker
        self._para_text.append(text)
        self._last_end = end
        self.segments_written += 1
        line = f"{speaker_label(speaker)}: {text}" if speaker else text
        self._srt.write(f"{self.segments_written}\n{format_srt_timestamp(start)} --> "
                        f"{format_srt_timestamp(end)}\n{line}\n\n")
        self._srt.flush()
        self._txt.write(f"[{format_timestamp(start)}] {line}\n")
        self._txt.flush()

    def _flush_paragraph(self) -> None:
        if not self._para_text or self._file is None:
            return
        body = " ".join(self._para_text)
        if self._para_speaker:
            body = f"**{speaker_label(self._para_speaker)}:** {body}"
        if self.timestamps:
            body = f"**[{format_timestamp(self._para_start)}]** {body}"
        self._file.write(body + "\n\n")
        self._file.flush()
        self._para_text = []

    def close(self, note: str | None = None) -> None:
        if self._file is None:
            return
        self._flush_paragraph()
        if self.segments_written == 0:
            self._file.write("_No se ha detectado voz en el archivo._\n\n")
        if note:
            self._file.write(f"---\n\n{note}\n")
        for f in (self._file, self._srt, self._txt):
            f.close()
        self._file = self._srt = self._txt = None
