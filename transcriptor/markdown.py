"""Escritura incremental de la transcripción en Markdown.

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


def _md_escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


class MarkdownWriter:
    def __init__(self, path: Path, timestamps: bool = True):
        self.path = path
        self.timestamps = timestamps
        self._file = None
        self._para_text: list[str] = []
        self._para_start = 0.0
        self._last_end: float | None = None
        self.segments_written = 0

    @property
    def is_open(self) -> bool:
        return self._file is not None

    def open(self, *, source: Path, duration: float | None, language: str,
             model: str, engine: str = "faster-whisper") -> None:
        self._file = open(self.path, "w", encoding="utf-8", newline="\n")
        rows = [
            ("Archivo", f"`{_md_escape_cell(source.name)}`"),
            ("Fecha", datetime.now().strftime("%d/%m/%Y %H:%M")),
            ("Duración", format_timestamp(duration) if duration else "desconocida"),
            ("Idioma", language),
            ("Modelo", f"{engine} `{model}`"),
        ]
        self._file.write(f"# Transcripción: {source.stem}\n\n")
        self._file.write("| | |\n|---|---|\n")
        for key, value in rows:
            self._file.write(f"| **{key}** | {value} |\n")
        self._file.write("\n---\n\n")
        self._file.flush()

    def add_segment(self, start: float, end: float, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if self._para_text and (
            (self._last_end is not None and start - self._last_end >= PARAGRAPH_GAP)
            or sum(len(t) + 1 for t in self._para_text) >= PARAGRAPH_MAX_CHARS
        ):
            self._flush_paragraph()
        if not self._para_text:
            self._para_start = start
        self._para_text.append(text)
        self._last_end = end
        self.segments_written += 1

    def _flush_paragraph(self) -> None:
        if not self._para_text or self._file is None:
            return
        body = " ".join(self._para_text)
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
        self._file.close()
        self._file = None
