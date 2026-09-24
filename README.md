# Transcriptor

Aplicación de escritorio para Windows (un único `Transcriptor.exe`) que transcribe
audios y vídeos de cualquier tamaño con [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
y guarda el resultado como Markdown en la carpeta **Descargas**.

Todo el proceso ocurre en el propio equipo: el audio no se envía a ningún servidor.

## Uso

1. Abre `Transcriptor.exe` (no necesita instalación ni tener Python).
2. Añade uno o varios archivos con **Añadir archivos…** o arrastrándolos a la ventana
   (también carpetas completas).
3. Elige el modelo, el idioma (o detección automática) y pulsa **Transcribir**.
4. La barra muestra el porcentaje, la posición del audio, el tiempo transcurrido,
   el tiempo restante estimado y la velocidad; debajo se ve el texto a medida que se
   genera.
5. Al terminar se crea `Descargas\<nombre> - transcripción.md` y puedes abrirlo
   directamente.

La primera vez que se usa un modelo se descarga desde Hugging Face (con barra de
progreso) y se guarda en `%LOCALAPPDATA%\Transcriptor\models`; a partir de ahí
funciona sin conexión.

| Modelo | Descarga | Comentario |
|---|---|---|
| tiny | 75 MB | Muy rápido, precisión baja |
| base | 145 MB | Rápido |
| small | 484 MB | **Recomendado en CPU** |
| medium | 1,5 GB | Más preciso y lento |
| large-v3-turbo | 1,6 GB | Muy preciso; ideal con GPU |
| large-v3 | 3,1 GB | Máxima precisión |

### Archivos grandes

El audio se decodifica en streaming y se procesa en bloques de ~20 minutos,
cortados en el punto más silencioso cercano al límite para no partir palabras. La
memoria usada no depende de la duración del archivo, y el `.md` se va escribiendo
en disco párrafo a párrafo: si cancelas (o algo falla) se conserva lo transcrito
hasta ese momento, con una nota indicando dónde se detuvo.

### Formato del Markdown

```markdown
# Transcripción: Reunión trimestral

| | |
|---|---|
| **Archivo** | `Reunión trimestral.mp4` |
| **Fecha** | 24/09/2026 17:30 |
| **Duración** | 01:02:15 |
| **Idioma** | Español (detectado, 98%) |
| **Modelo** | faster-whisper `small` |

---

**[00:00:00]** Buenos días a todos, gracias por venir. Vamos a repasar…

**[00:00:42]** Ahora paso la palabra a Marta…
```

Las marcas de tiempo por párrafo son opcionales.

### GPU NVIDIA

En *Procesar con* puedes elegir la GPU. Necesita tener instalados CUDA 12 y cuDNN 9
(no se incluyen en el `.exe` para no multiplicar su tamaño). Si no están disponibles,
la aplicación avisa y continúa con la CPU.

## Descargar el ejecutable

Cada push compila el `.exe` en GitHub Actions (workflow *Build Windows exe*), lo
prueba con una transcripción real y lo publica como artefacto
**Transcriptor-windows**. Para publicar una *Release* con el `.exe`, crea una
etiqueta `v*` (por ejemplo `v1.0.0`) o lanza el workflow a mano (*Run workflow*)
indicando la versión en el campo *release*: se crea la etiqueta y la release solo
si el ejecutable supera el autotest.

## Compilar en local

En Windows con Python 3.12 de python.org:

```bat
build.bat
```

El resultado queda en `dist\Transcriptor.exe`. Para ejecutar desde el código fuente:

```bat
pip install -r requirements.txt
python main.py
```

Autotest del ejecutable (lo usa la CI): `Transcriptor.exe --self-test [--model tiny] [--report informe.txt]`.

## Estructura

- `main.py` — punto de entrada.
- `transcriptor/gui.py` — interfaz (tkinter).
- `transcriptor/engine.py` — descarga/carga del modelo y transcripción en segundo plano.
- `transcriptor/audio.py` — decodificación en streaming y troceado por silencios (PyAV).
- `transcriptor/markdown.py` — escritura incremental del `.md`.
- `transcriptor/paths.py` — carpeta Descargas real de Windows, ajustes y registros.
- `Transcriptor.spec` — configuración de PyInstaller (un solo archivo, sin consola).

Los registros de errores están en `%LOCALAPPDATA%\Transcriptor\transcriptor.log`.
