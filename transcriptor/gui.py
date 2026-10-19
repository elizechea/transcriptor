"""Interfaz gráfica de Transcriptor (tkinter)."""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk

from . import APP_NAME, __version__
from .audio import AUDIO_EXTENSIONS, MEDIA_EXTENSIONS, VIDEO_EXTENSIONS
from .engine import DEFAULT_DEVICE, DEVICES, LANGUAGES, MODELS, Options, TranscriptionJob
from .markdown import format_timestamp
from .paths import downloads_dir, load_settings, open_in_explorer, save_settings

log = logging.getLogger(__name__)

PREVIEW_MAX_LINES = 3000
# Los puntos de Tk en macOS se ven más pequeños que en Windows.
_FONT_SCALE = 1.3 if sys.platform == "darwin" else 1.0


def _pt(size: int) -> int:
    return round(size * _FONT_SCALE)


def resource_path(name: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / "assets" / name


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.events: queue.Queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.files: list[Path] = []
        self.outputs: list[Path] = []
        self.errors: list[tuple[Path, str]] = []
        self.out_dir = downloads_dir()
        self.file_started_at = 0.0
        self.settings = load_settings()

        self._setup_window()
        self._setup_style()
        self._build()
        self._enable_drag_and_drop()
        self._set_running(False)
        # Tamaño natural según fuentes y escala DPI de Windows, no en píxeles fijos.
        self.root.update_idletasks()
        width, height = self.root.winfo_reqwidth(), self.root.winfo_reqheight()
        self.root.minsize(width, height)
        extra = max(0, min(260, self.root.winfo_screenheight() - height - 120))
        self.root.geometry(f"{width}x{height + extra}")
        self.root.after(100, self._poll_events)

    # ------------------------------------------------------------------ UI
    def _setup_window(self) -> None:
        self.root.title(APP_NAME)
        try:
            if sys.platform == "win32":
                self.root.iconbitmap(default=str(resource_path("icon.ico")))
            else:
                self.root.iconphoto(True, tk.PhotoImage(file=str(resource_path("icon.png"))))
        except Exception:
            pass
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_style(self) -> None:
        style = ttk.Style(self.root)
        for theme in ("aqua", "vista", "clam"):  # nativo en macOS y Windows
            if theme in style.theme_names():
                style.theme_use(theme)
                break
        family = "Segoe UI" if sys.platform == "win32" else tkfont.nametofont("TkDefaultFont").actual("family")
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            tkfont.nametofont(name).configure(family=family, size=_pt(10))
        self.font_family = family
        style.configure("Title.TLabel", font=(family, _pt(20), "bold"), foreground="#1f4e79")
        style.configure("Subtitle.TLabel", font=(family, _pt(10)), foreground="#555555")
        style.configure("Section.TLabelframe.Label", font=(family, _pt(11), "bold"),
                        foreground="#1f4e79")
        style.configure("Hint.TLabel", foreground="#666666")
        style.configure("Status.TLabel", font=(family, _pt(10), "italic"), foreground="#444444")
        style.configure("Big.TLabel", font=(family, _pt(16), "bold"), foreground="#1f4e79")
        style.configure("Primary.TButton", font=(family, _pt(11), "bold"), padding=(18, 8))
        style.configure("TButton", padding=(10, 4))
        style.configure("Horizontal.TProgressbar", thickness=22)

    def _build(self) -> None:
        main = ttk.Frame(self.root, padding=(16, 12, 16, 12))
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)

        header = ttk.Frame(main)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(header, text="🎙  " + APP_NAME, style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Transcribe audio y vídeo a texto con Whisper, en tu propio equipo "
                 "y sin enviar nada a Internet.",
            style="Subtitle.TLabel",
        ).pack(anchor="w")

        # 1. Archivos -------------------------------------------------------
        files = ttk.LabelFrame(main, text=" 1. Archivos ", style="Section.TLabelframe",
                               padding=10)
        files.grid(row=1, column=0, sticky="ew", pady=4)
        files.columnconfigure(0, weight=1)
        list_frame = ttk.Frame(files)
        list_frame.grid(row=0, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        self.listbox = tk.Listbox(list_frame, height=4, selectmode="extended",
                                  activestyle="none", borderwidth=1, relief="solid",
                                  highlightthickness=0, font=(self.font_family, _pt(10)),
                                  background="#ffffff", foreground="#222222")
        self.listbox.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scroll.set)
        self.listbox.bind("<Double-Button-1>", lambda _e: self._add_files())
        self.listbox.bind("<Delete>", lambda _e: self._remove_selected())

        buttons = ttk.Frame(files)
        buttons.grid(row=0, column=1, sticky="n", padx=(10, 0))
        self.btn_add = ttk.Button(buttons, text="➕ Añadir archivos…", command=self._add_files)
        self.btn_add.pack(fill="x")
        self.btn_remove = ttk.Button(buttons, text="Quitar", command=self._remove_selected)
        self.btn_remove.pack(fill="x", pady=4)
        self.btn_clear = ttk.Button(buttons, text="Vaciar lista", command=self._clear_files)
        self.btn_clear.pack(fill="x")
        self.files_hint = ttk.Label(
            files, style="Hint.TLabel",
            text="Añade uno o varios archivos (MP3, WAV, M4A, MP4, MKV, MOV…). "
                 "Sin límite de tamaño.",
        )
        self.files_hint.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # 2. Opciones -------------------------------------------------------
        opts = ttk.LabelFrame(main, text=" 2. Opciones ", style="Section.TLabelframe",
                              padding=10)
        opts.grid(row=2, column=0, sticky="ew", pady=4)
        opts.columnconfigure(1, weight=1)

        s = self.settings
        self.model_var = tk.StringVar(value=self._label_for(MODELS, s.get("model"), "small"))
        self.lang_var = tk.StringVar(value=self._label_for(LANGUAGES, s.get("language")))
        # Un ajuste guardado cuando aún no existía la opción recomendada (p. ej. la
        # GPU del Mac) no fue una elección: se propone la recomendada.
        saved_device = (s.get("device")
                        if DEFAULT_DEVICE in s.get("device_choices", [DEFAULT_DEVICE])
                        else None)
        self.device_var = tk.StringVar(
            value=self._label_for(DEVICES, saved_device, DEFAULT_DEVICE))
        self.timestamps_var = tk.BooleanVar(value=s.get("timestamps", True))
        self.vad_var = tk.BooleanVar(value=s.get("vad", True))

        self.inputs: list[tk.Widget] = []
        rows = [
            ("Modelo:", self.model_var, [m[1] for m in MODELS]),
            ("Idioma del audio:", self.lang_var, [l[1] for l in LANGUAGES]),
            ("Procesar con:", self.device_var, [d[1] for d in DEVICES]),
        ]
        for r, (label, var, values) in enumerate(rows):
            ttk.Label(opts, text=label).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 10))
            combo = ttk.Combobox(opts, textvariable=var, values=values, state="readonly",
                                 width=48)
            combo.grid(row=r, column=1, sticky="w", pady=3)
            self.inputs.append(combo)
        checks = ttk.Frame(opts)
        checks.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        c1 = ttk.Checkbutton(checks, text="Incluir marcas de tiempo", variable=self.timestamps_var)
        c1.pack(side="left")
        c2 = ttk.Checkbutton(checks, text="Omitir silencios y música (recomendado)",
                             variable=self.vad_var)
        c2.pack(side="left", padx=(20, 0))
        self.inputs += [c1, c2]
        ttk.Label(
            opts, style="Hint.TLabel",
            text="El modelo se descarga automáticamente la primera vez que se usa.",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        dest = ttk.Frame(main)
        dest.grid(row=3, column=0, sticky="ew", pady=(6, 4))
        ttk.Label(dest, text="📁 La transcripción (.md, .srt y .txt) se guardará en:").pack(side="left")
        ttk.Label(dest, text=str(self.out_dir), font=(self.font_family, _pt(10), "bold")).pack(
            side="left", padx=6)
        ttk.Button(dest, text="Abrir carpeta",
                   command=lambda: open_in_explorer(self.out_dir)).pack(side="right")

        # 3. Progreso -------------------------------------------------------
        prog = ttk.LabelFrame(main, text=" 3. Progreso ", style="Section.TLabelframe",
                              padding=10)
        prog.grid(row=4, column=0, sticky="nsew", pady=4)
        main.rowconfigure(4, weight=1)
        prog.columnconfigure(0, weight=1)
        prog.rowconfigure(4, weight=1)

        top = ttk.Frame(prog)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        self.lbl_file = ttk.Label(top, text="Esperando archivos…")
        self.lbl_file.grid(row=0, column=0, sticky="w")
        self.lbl_percent = ttk.Label(top, text="", style="Big.TLabel")
        self.lbl_percent.grid(row=0, column=1, sticky="e")

        self.progress = ttk.Progressbar(prog, orient="horizontal", mode="determinate",
                                        maximum=1000)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(4, 4))
        self.lbl_times = ttk.Label(prog, text="", style="Hint.TLabel")
        self.lbl_times.grid(row=2, column=0, sticky="w")
        self.lbl_status = ttk.Label(prog, text="", style="Status.TLabel")
        self.lbl_status.grid(row=3, column=0, sticky="w", pady=(2, 6))

        text_frame = ttk.Frame(prog)
        text_frame.grid(row=4, column=0, sticky="nsew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        self.preview = tk.Text(text_frame, height=5, width=90, wrap="word", state="disabled",
                               font=(self.font_family, _pt(10)), relief="solid", borderwidth=1,
                               padx=8, pady=6, background="#fbfbfb",
                               foreground="#222222")
        self.preview.grid(row=0, column=0, sticky="nsew")
        self.preview.tag_configure("ts", foreground="#1f4e79")
        self.preview.tag_configure("info", foreground="#777777", font=(self.font_family, _pt(9), "italic"))
        pscroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.preview.yview)
        pscroll.grid(row=0, column=1, sticky="ns")
        self.preview.configure(yscrollcommand=pscroll.set)

        # Botones inferiores --------------------------------------------------
        bottom = ttk.Frame(main)
        bottom.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        self.btn_open = ttk.Button(bottom, text="📄 Abrir transcripción",
                                   command=self._open_last, state="disabled")
        self.btn_open.pack(side="left")
        ttk.Label(bottom, text=f"v{__version__}", style="Hint.TLabel").pack(side="left", padx=10)
        self.btn_start = ttk.Button(bottom, text="▶  Transcribir", style="Primary.TButton",
                                    command=self._start)
        self.btn_start.pack(side="right")
        self.btn_cancel = ttk.Button(bottom, text="Cancelar", command=self._cancel)
        self.btn_cancel.pack(side="right", padx=8)

    @staticmethod
    def _label_for(options, key, default=None) -> str:
        """Etiqueta visible de `key`; si no existe (ajuste antiguo), la de `default`."""
        for wanted in (key, default):
            for k, label, *_rest in options:
                if k == wanted:
                    return label
        return options[0][1]

    @staticmethod
    def _key_for(options, label):
        for k, lbl, *_rest in options:
            if lbl == label:
                return k
        return options[0][0]

    def _enable_drag_and_drop(self) -> None:
        if sys.platform != "win32":
            return
        try:
            import windnd

            def dropped(paths):
                # Se difiere al bucle de Tk para no bloquear el mensaje de Windows.
                decoded = [p.decode("mbcs") if isinstance(p, bytes) else p for p in paths]
                self.root.after(0, lambda: self._add_paths([Path(p) for p in decoded]))

            windnd.hook_dropfiles(self.root, func=dropped, force_unicode=True)
            self.files_hint.configure(
                text="Arrastra aquí archivos o carpetas, o pulsa «Añadir archivos…» "
                     "(MP3, WAV, M4A, MP4, MKV, MOV…). Sin límite de tamaño.")
        except Exception:
            log.info("Arrastrar y soltar no disponible", exc_info=True)

    # ------------------------------------------------------------ archivos
    def _add_files(self) -> None:
        if self.worker is not None:
            return
        pattern = " ".join("*" + e for e in MEDIA_EXTENSIONS)
        selected = filedialog.askopenfilenames(
            parent=self.root,
            title="Selecciona audios o vídeos",
            initialdir=self.settings.get("last_dir") or str(Path.home()),
            filetypes=[
                ("Audio y vídeo", pattern),
                ("Audio", " ".join("*" + e for e in AUDIO_EXTENSIONS)),
                ("Vídeo", " ".join("*" + e for e in VIDEO_EXTENSIONS)),
                ("Todos los archivos", "*.*"),
            ],
        )
        if selected:
            self.settings["last_dir"] = str(Path(selected[0]).parent)
            self._add_paths([Path(p) for p in selected])

    def _add_paths(self, paths: list[Path]) -> None:
        if self.worker is not None:
            return
        expanded: list[Path] = []
        for p in paths:
            if p.is_dir():
                expanded += sorted(f for f in p.rglob("*")
                                   if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS)
            elif p.is_file():
                expanded.append(p)
        for p in expanded:
            if p not in self.files:
                self.files.append(p)
                self.listbox.insert("end", f"{p.name}   ({_human_size(p.stat().st_size)})")
        self._update_file_label()

    def _remove_selected(self) -> None:
        if self.worker is not None:
            return
        for index in reversed(self.listbox.curselection()):
            self.listbox.delete(index)
            del self.files[index]
        self._update_file_label()

    def _clear_files(self) -> None:
        if self.worker is not None:
            return
        self.listbox.delete(0, "end")
        self.files.clear()
        self._update_file_label()

    def _update_file_label(self) -> None:
        n = len(self.files)
        self.lbl_file.configure(
            text="Esperando archivos…" if n == 0 else
            f"{n} archivo{'s' if n != 1 else ''} listo{'s' if n != 1 else ''} para transcribir")

    # ------------------------------------------------------------- trabajo
    def _current_options(self) -> Options:
        return Options(
            model=self._key_for(MODELS, self.model_var.get()),
            language=self._key_for(LANGUAGES, self.lang_var.get()),
            device=self._key_for(DEVICES, self.device_var.get()),
            timestamps=self.timestamps_var.get(),
            vad=self.vad_var.get(),
        )

    def _start(self) -> None:
        if self.worker is not None:
            return
        if not self.files:
            self._add_files()
            if not self.files:
                return
        options = self._current_options()
        self.settings.update(model=options.model, language=options.language,
                             device=options.device, timestamps=options.timestamps,
                             vad=options.vad, device_choices=[d[0] for d in DEVICES])
        save_settings(self.settings)

        self.outputs.clear()
        self.errors.clear()
        self.cancel_event.clear()
        self._clear_preview()
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.lbl_percent.configure(text="")
        self.lbl_times.configure(text="")
        self._set_running(True)
        job = TranscriptionJob(list(self.files), options, self.out_dir,
                               self._emit, self.cancel_event)
        self.worker = threading.Thread(target=job.run, daemon=True)
        self.worker.start()

    def _emit(self, kind: str, **data) -> None:
        self.events.put((kind, data))

    def _cancel(self) -> None:
        if self.worker is None:
            return
        if messagebox.askyesno(APP_NAME, "¿Cancelar la transcripción en curso?\n\n"
                               "Se conservará en Descargas el texto transcrito hasta ahora.",
                               parent=self.root):
            self.cancel_event.set()
            self.lbl_status.configure(text="Cancelando…")
            self.btn_cancel.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        for w in (self.btn_add, self.btn_remove, self.btn_clear, self.btn_start):
            w.configure(state=state)
        for w in self.inputs:
            w.configure(state="disabled" if running else
                        ("readonly" if isinstance(w, ttk.Combobox) else "normal"))
        self.btn_cancel.configure(state="normal" if running else "disabled")

    # ------------------------------------------------------------- eventos
    def _poll_events(self) -> None:
        try:
            for _ in range(500):
                kind, data = self.events.get_nowait()
                getattr(self, f"_on_{kind}")(**data)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _on_status(self, text: str) -> None:
        self.lbl_status.configure(text=text)

    def _on_warning(self, text: str) -> None:
        self._append_info(text.split("\n")[0])
        messagebox.showwarning(APP_NAME, text, parent=self.root)

    def _on_download(self, done: int, total: int) -> None:
        if total <= 0:
            self._indeterminate()
            self.lbl_times.configure(text="")
            return
        if str(self.progress.cget("mode")) != "determinate":
            self.progress.stop()
            self.progress.configure(mode="determinate")
        fraction = min(done / total, 0.99)
        self.progress["value"] = fraction * 1000
        self.lbl_percent.configure(text=f"{fraction:.0%}")
        self.lbl_times.configure(
            text=f"Descargado {done / 1e6:,.0f} MB de ~{total / 1e6:,.0f} MB".replace(",", "."))

    def _on_file_start(self, index: int, total: int, path: Path) -> None:
        self.file_started_at = time.monotonic()
        prefix = f"Archivo {index + 1} de {total}: " if total > 1 else ""
        self.lbl_file.configure(text=f"{prefix}{path.name}")
        self.lbl_percent.configure(text="0%")
        self.lbl_times.configure(text="")
        self._append_info(f"— {path.name} —")
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(index)
        self.listbox.see(index)

    def _on_language(self, code: str, probability: float) -> None:
        from .engine import LANGUAGE_NAMES
        self._append_info(f"Idioma detectado: {LANGUAGE_NAMES.get(code, code)} "
                          f"({probability:.0%})")

    def _on_progress(self, fraction: float | None, position: float,
                     duration: float | None) -> None:
        elapsed = time.monotonic() - self.file_started_at
        if fraction is None:
            self._indeterminate()
            self.lbl_percent.configure(text="")
            self.lbl_times.configure(
                text=f"Audio procesado: {format_timestamp(position)}   ·   "
                     f"Tiempo transcurrido: {format_timestamp(elapsed)}")
            return
        if str(self.progress.cget("mode")) != "determinate":
            self.progress.stop()
            self.progress.configure(mode="determinate")
        self.progress["value"] = fraction * 1000
        self.lbl_percent.configure(text=f"{fraction:.0%}")
        self.root.title(f"{fraction:.0%} — {APP_NAME}")
        parts = [f"Audio {format_timestamp(position)} / {format_timestamp(duration or 0)}",
                 f"Transcurrido {format_timestamp(elapsed)}"]
        if fraction > 0.005 and elapsed > 3:
            remaining = elapsed * (1 - fraction) / fraction
            parts.append(f"Restante ≈ {format_timestamp(remaining)}")
            if elapsed > 0:
                parts.append(f"Velocidad ×{position / elapsed:.1f}")
        self.lbl_times.configure(text="   ·   ".join(parts))

    def _on_segment(self, start: float, end: float, text: str) -> None:
        if not text:
            return
        self.preview.configure(state="normal")
        self.preview.insert("end", f"[{format_timestamp(start)}] ", "ts")
        self.preview.insert("end", text + "\n")
        self._trim_preview()
        self.preview.configure(state="disabled")
        self.preview.see("end")

    def _on_file_done(self, path: Path, output: Path) -> None:
        self.outputs.append(output)
        self.btn_open.configure(state="normal")
        self._append_info(f"✔ Guardado: {output.name}")

    def _on_file_error(self, path: Path, message: str) -> None:
        self.errors.append((path, message))
        self._append_info(f"✖ Error en {path.name}: {message}")

    def _on_cancelled(self) -> None:
        self._append_info("Transcripción cancelada.")

    def _on_fatal(self, message: str) -> None:
        self.errors.append((Path("-"), message))
        messagebox.showerror(APP_NAME, message, parent=self.root)

    def _on_finished(self) -> None:
        self.worker = None
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self._set_running(False)
        self.root.title(APP_NAME)
        cancelled = self.cancel_event.is_set()
        if cancelled:
            self.lbl_status.configure(text="Cancelado. El texto parcial se ha guardado en Descargas.")
            return
        if self.outputs:
            self.progress["value"] = 1000
            self.lbl_percent.configure(text="100%")
        self.lbl_status.configure(
            text=f"Completado: {len(self.outputs)} transcripción(es), {len(self.errors)} error(es).")
        if self.errors and not self.outputs:
            if self.errors[0][0] != Path("-"):
                messagebox.showerror(APP_NAME, "No se pudo transcribir ningún archivo:\n\n" +
                                     "\n".join(f"• {p.name}: {m}" for p, m in self.errors),
                                     parent=self.root)
            return
        if not self.outputs:
            return
        msg = (f"Transcripción guardada en Descargas:\n\n{self.outputs[0].name}"
               if len(self.outputs) == 1 else
               f"Se han guardado {len(self.outputs)} transcripciones en Descargas.")
        if self.errors:
            msg += "\n\nArchivos con errores:\n" + "\n".join(
                f"• {p.name}: {m}" for p, m in self.errors)
        question = "¿Quieres abrirla ahora?" if len(self.outputs) == 1 else "¿Abrir la carpeta?"
        if messagebox.askyesno(APP_NAME, f"{msg}\n\n{question}", parent=self.root):
            self._open_last()

    # -------------------------------------------------------------- varios
    def _indeterminate(self) -> None:
        if str(self.progress.cget("mode")) != "indeterminate":
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)

    def _append_info(self, text: str) -> None:
        self.preview.configure(state="normal")
        self.preview.insert("end", text + "\n", "info")
        self._trim_preview()
        self.preview.configure(state="disabled")
        self.preview.see("end")

    def _trim_preview(self) -> None:
        lines = int(self.preview.index("end-1c").split(".")[0])
        if lines > PREVIEW_MAX_LINES:
            self.preview.delete("1.0", f"{lines - PREVIEW_MAX_LINES}.0")

    def _clear_preview(self) -> None:
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.configure(state="disabled")

    def _open_last(self) -> None:
        if len(self.outputs) == 1 and self.outputs[0].exists():
            open_in_explorer(self.outputs[0])
        elif self.outputs:
            open_in_explorer(self.out_dir)

    def _on_close(self) -> None:
        if self.worker is not None and not messagebox.askyesno(
                APP_NAME, "Hay una transcripción en curso. ¿Quieres cancelarla y salir?",
                parent=self.root):
            return
        self.cancel_event.set()
        save_settings(self.settings)
        self.root.destroy()


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def run() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()
