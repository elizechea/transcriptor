# Transcriptor

Aplicación de escritorio para **Windows** (un único `Transcriptor.exe`) y **macOS
con chip Apple M1/M2/M3/M4…** (`Transcriptor.app`, macOS 13 Ventura o posterior) que transcribe
audios y vídeos de cualquier tamaño con [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
y guarda el resultado como Markdown en la carpeta **Descargas**.

Todo el proceso ocurre en el propio equipo: el audio no se envía a ningún servidor.

## Uso

1. Abre `Transcriptor.exe` en Windows o `Transcriptor.app` en Mac (no necesita
   instalación ni tener Python; ver [Instalar en Mac](#instalar-en-mac)).
2. Añade uno o varios archivos con **Añadir archivos…** (en Windows también
   arrastrándolos a la ventana, incluso carpetas completas).
3. Elige el modelo, el idioma (o detección automática) y pulsa **Transcribir**.
4. La barra muestra el porcentaje, la posición del audio, el tiempo transcurrido,
   el tiempo restante estimado y la velocidad; debajo se ve el texto a medida que se
   genera.
5. Al terminar se crea `<nombre> - transcripción.md` en la carpeta Descargas y
   puedes abrirlo directamente.

La primera vez que se usa un modelo se descarga desde Hugging Face (con barra de
progreso) y se guarda en `%LOCALAPPDATA%\Transcriptor\models` (Windows) o en
`~/Library/Application Support/Transcriptor/models` (Mac); a partir de ahí
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

### Rendimiento en segundo plano

Windows 11 frena los programas cuya ventana no está activa (limitación de energía
*EcoQoS*: menos frecuencia y núcleos de eficiencia), y macOS hace lo mismo con
*App Nap*. Transcriptor desactiva ambos para su proceso, así que puedes dejar la
ventana minimizada o seguir trabajando en otra cosa sin que la transcripción se
ralentice. Mientras transcribe, el equipo tampoco entra en reposo por inactividad
(la pantalla sí puede apagarse).

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

### Mac con chip M

La versión de Mac es nativa para Apple Silicon (arm64). Por defecto transcribe en
la **GPU** del chip M con [MLX](https://github.com/ml-explore/mlx) (mlx-whisper),
ya que faster-whisper solo puede usar la CPU en Mac. En un M5 Pro, `large-v3-turbo`
va unas 5 veces más rápido en la GPU que en la CPU, así que es la opción
recomendada en cualquier chip M.

- La GPU necesita **macOS 14 Sonoma** o posterior; en macOS 13 la app avisa y usa
  la CPU. En *Procesar con* también se puede elegir la CPU a mano.
- Los modelos MLX son distintos de los de CPU: el primer uso de cada modelo en la
  GPU lo descarga de nuevo (mismo tamaño).
- En la GPU el texto aparece en bloques de 5 minutos de audio (la barra de
  progreso avanza de forma continua). Si cancelas, se conserva el texto de los
  bloques ya terminados.

### Instalar en Mac

1. Descarga `Transcriptor-macOS.dmg` de la *Release*, ábrelo y arrastra
   **Transcriptor** a **Aplicaciones**.
2. La app no está firmada con un certificado de Apple Developer, así que la primera
   vez macOS la bloquea. Ábrela una vez, cierra el aviso y ve a
   **Ajustes del Sistema → Privacidad y seguridad → «Abrir igualmente»**.
   También puedes ejecutar en Terminal:
   `xattr -dr com.apple.quarantine /Applications/Transcriptor.app`
3. La primera vez que guarde una transcripción, macOS pedirá permiso para acceder
   a la carpeta Descargas: acéptalo.

### GPU NVIDIA (solo Windows)

En *Procesar con* puedes elegir la GPU. Necesita tener instalados CUDA 12 y cuDNN 9
(no se incluyen en el `.exe` para no multiplicar su tamaño). Si no están disponibles,
la aplicación avisa y continúa con la CPU.

## Descargar

Las versiones publicadas están en
[Releases](https://github.com/elizechea/transcriptor/releases): `Transcriptor.exe`
para Windows y `Transcriptor-macOS.dmg` para Mac.

Cada push compila ambas en GitHub Actions (workflow *Build*), las prueba con una
transcripción real y las publica como artefactos **Transcriptor-windows** y
**Transcriptor-macOS**. Para publicar una *Release*, crea una etiqueta `v*`
(por ejemplo `v1.1.0`) o lanza el workflow a mano (*Run workflow*) indicando la
versión en el campo *release*: la etiqueta y la release solo se crean si las dos
versiones superan el autotest.

## Compilar en local

En Windows con Python 3.12 de python.org:

```bat
build.bat
```

El resultado queda en `dist\Transcriptor.exe`.

En un Mac con chip M, con Python 3.12 de python.org:

```bash
./build_mac.sh
```

Genera `dist/Transcriptor.app` y `dist/Transcriptor-macOS.dmg`.

Para ejecutar desde el código fuente:

```bat
pip install -r requirements.txt
python main.py
```

Autotest del ejecutable (lo usa la CI): `Transcriptor.exe --self-test [--model tiny] [--report informe.txt]`
(en Mac: `Transcriptor.app/Contents/MacOS/Transcriptor --self-test …`).

## Estructura

- `main.py` — punto de entrada.
- `transcriptor/gui.py` — interfaz (tkinter).
- `transcriptor/engine.py` — descarga/carga del modelo y transcripción en segundo plano.
- `transcriptor/audio.py` — decodificación en streaming y troceado por silencios (PyAV).
- `transcriptor/markdown.py` — escritura incremental del `.md`.
- `transcriptor/paths.py` — carpeta Descargas real de Windows, ajustes y registros.
- `transcriptor/power.py` — evita la limitación en segundo plano y el reposo.
- `Transcriptor.spec` — configuración de PyInstaller (`.exe` de un solo archivo en
  Windows, `.app` en macOS).

Los registros de errores están en `%LOCALAPPDATA%\Transcriptor\transcriptor.log`
(Windows) o `~/Library/Application Support/Transcriptor/transcriptor.log` (Mac).
