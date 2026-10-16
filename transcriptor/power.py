"""Evita que el sistema frene la transcripción cuando la ventana no está activa.

* Windows 10/11 aplica "EcoQoS" (limitación de energía) a los procesos cuya
  ventana está en segundo plano o minimizada: baja la frecuencia de la CPU y
  mueve el trabajo a los núcleos de eficiencia. Se desactiva para este proceso
  con SetProcessInformation(ProcessPowerThrottling).
* macOS aplica "App Nap" a las apps ocultas. Se desactiva en Info.plist
  (NSAppSleepDisabled) y, mientras se transcribe, con `caffeinate`.
* En Linux no hay limitación equivalente; durante la transcripción se pide a
  systemd-logind que no suspenda el equipo (`systemd-inhibit`).
* En todos los casos, durante la transcripción se impide que el equipo entre en
  reposo por inactividad (la pantalla sí puede apagarse).
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys

from .paths import system_env

log = logging.getLogger(__name__)

# Windows: winbase.h / processthreadsapi.h
_PROCESS_POWER_THROTTLING = 4  # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
_PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
_PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
_PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x4
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def disable_background_throttling() -> bool:
    """Pide a Windows que no limite la velocidad de este proceso (EcoQoS)."""
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
        _fields_ = [
            ("Version", wintypes.ULONG),
            ("ControlMask", wintypes.ULONG),
            ("StateMask", wintypes.ULONG),
        ]

    mask = (_PROCESS_POWER_THROTTLING_EXECUTION_SPEED
            | _PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION)
    # ControlMask indica qué se gestiona; StateMask = 0 lo desactiva explícitamente.
    state = PROCESS_POWER_THROTTLING_STATE(_PROCESS_POWER_THROTTLING_CURRENT_VERSION, mask, 0)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetProcessInformation.restype = wintypes.BOOL
    ok = bool(kernel32.SetProcessInformation(
        kernel32.GetCurrentProcess(), _PROCESS_POWER_THROTTLING,
        ctypes.byref(state), ctypes.sizeof(state)))
    if not ok:
        # Windows 7/8 no conocen esta opción: no es un error grave.
        log.info("SetProcessInformation(PowerThrottling) falló: %s", ctypes.get_last_error())
    return ok


class KeepAwake:
    """Mientras está activo impide el reposo del equipo y el App Nap de macOS.

    En Windows el estado va ligado al hilo que llama, así que debe usarse
    desde el hilo que hace el trabajo.
    """

    def __init__(self) -> None:
        self._helper: subprocess.Popen | None = None

    def __enter__(self) -> "KeepAwake":
        try:
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(
                    _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
            elif sys.platform == "darwin":
                # -i: sin reposo por inactividad; -w: termina si esta app muere.
                self._helper = subprocess.Popen(
                    ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif shutil.which("systemd-inhibit"):
                # El bloqueo dura lo que el `tail`, que acaba si esta app muere.
                # Va en su propio grupo de procesos para poder terminar ambos.
                self._helper = subprocess.Popen(
                    ["systemd-inhibit", "--what=idle:sleep", "--who=Transcriptor",
                     "--why=Transcribiendo", "--mode=block",
                     "tail", f"--pid={os.getpid()}", "-f", "/dev/null"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    env=system_env(), start_new_session=True)
        except Exception:
            log.info("No se pudo evitar el reposo del equipo", exc_info=True)
        return self

    def __exit__(self, *_exc) -> None:
        try:
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
            elif self._helper is not None:
                if sys.platform == "darwin":
                    self._helper.terminate()
                else:
                    os.killpg(self._helper.pid, signal.SIGTERM)
                self._helper.wait(timeout=5)
                self._helper = None
        except Exception:
            log.info("No se pudo restaurar el estado de energía", exc_info=True)
