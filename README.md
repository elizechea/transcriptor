# Transcriptor

Aplicación de escritorio para **Windows** (un único `Transcriptor.exe`), **macOS
con chip Apple M1/M2/M3/M4…** (`Transcriptor.app`, macOS 13 Ventura o posterior) y
**Linux x86_64** (un único ejecutable `Transcriptor`) que transcribe
audios y vídeos de cualquier tamaño con [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
y guarda el resultado como Markdown en la carpeta **Descargas**.

Todo el proceso ocurre en el propio equipo: el audio no se envía a ningún servidor.

## Primera ejecución: el sistema la bloquea

La aplicación no está firmada con un certificado de pago de Apple ni de Microsoft,
así que la primera vez el sistema la bloquea. Solo hay que autorizarla una vez.

### Mac: «No se ha abierto "Transcriptor.app"»

1. Descarga `Transcriptor-macOS.dmg` de la *Release*, ábrelo y arrastra
   **Transcriptor** a **Aplicaciones**.
2. Abre la app. Aparece el aviso *«Apple no ha podido verificar que
   "Transcriptor.app" no contenga software malicioso»*: pulsa **OK / Hecho**
   (**no** «Trasladar a la papelera»).
3. Ve a **Ajustes del Sistema → Privacidad y seguridad**, baja hasta la sección
   **Seguridad** y pulsa **Abrir igualmente** junto al mensaje sobre Transcriptor.
4. Vuelve a abrir la app, pulsa **Abrir igualmente** en el nuevo aviso y confirma
   con tu contraseña o Touch ID.

Alternativa en Terminal (hace lo mismo en un paso):

```bash
xattr -dr com.apple.quarantine /Applications/Transcriptor.app
```

La primera vez que guarde una transcripción, macOS pedirá permiso para acceder a la
carpeta Descargas: acéptalo.

### Windows: «Windows protegió su PC»

1. Al abrir `Transcriptor.exe` aparece la pantalla azul de **SmartScreen**.
2. Pulsa **Más información** y después **Ejecutar de todas formas**.

Si el navegador avisa al descargarlo de que «no se descarga habitualmente», elige
**Conservar** (en Edge: menú **…** → **Conservar** → **Mostrar más** →
**Conservar de todos modos**). Si Windows lo bloquea sin ofrecer *Ejecutar de todas
formas*, haz clic derecho en el `.exe` → **Propiedades**, marca **Desbloquear**
abajo del todo y pulsa **Aceptar**.

### Linux

No hay aviso, pero el archivo descargado no tiene permiso de ejecución. Descomprime el
`.tar.gz` (conserva el permiso) y abre `Transcriptor`:

```bash
tar -xzf Transcriptor-linux-x86_64.tar.gz
./Transcriptor
```

Necesita una distribución con glibc 2.35 o posterior (Ubuntu 22.04, Debian 12,
Fedora 36 o más recientes).

## Uso

1. Abre `Transcriptor.exe` en Windows, `Transcriptor.app` en Mac o `Transcriptor` en
   Linux (no necesita
   instalación ni tener Python; ver [Primera ejecución](#primera-ejecución-el-sistema-la-bloquea)).
2. Añade uno o varios archivos con **Añadir archivos…** (en Windows también
   arrastrándolos a la ventana, incluso carpetas completas).
3. Elige el modelo, el idioma (o detección automática) y pulsa **Transcribir**.
4. La barra muestra el porcentaje, la posición del audio, el tiempo transcurrido,
   el tiempo restante estimado y la velocidad; debajo se ve el texto a medida que se
   genera.
5. Al terminar se crea `<nombre> - transcripción.md` en la carpeta Descargas y
   puedes abrirlo directamente.

La primera vez que se usa un modelo se descarga desde Hugging Face (con barra de
progreso) y se guarda en `%LOCALAPPDATA%\Transcriptor\models` (Windows),
`~/Library/Application Support/Transcriptor/models` (Mac) o
`~/.local/share/Transcriptor/models` (Linux); a partir de ahí
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
(la pantalla sí puede apagarse); en Linux esto se pide con `systemd-inhibit`.

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

### GPU NVIDIA (Windows y Linux)

En *Procesar con* puedes elegir la GPU. Necesita tener instalados CUDA 12 y cuDNN 9
(no se incluyen en el ejecutable para no multiplicar su tamaño). Si no están disponibles,
la aplicación avisa y continúa con la CPU.

## Descargar

Las versiones publicadas están en
[Releases](https://github.com/elizechea/transcriptor/releases): `Transcriptor.exe`
para Windows, `Transcriptor-macOS.dmg` para Mac y `Transcriptor-linux-x86_64.tar.gz`
para Linux.

Cada push compila las tres en GitHub Actions (workflow *Build*), las prueba con una
transcripción real y las publica como artefactos **Transcriptor-windows**,
**Transcriptor-macOS** y **Transcriptor-linux**. Para publicar una *Release*, crea una etiqueta `v*`
(por ejemplo `v1.1.0`) o lanza el workflow a mano (*Run workflow*) indicando la
versión en el campo *release*: la etiqueta y la release solo se crean si las tres
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

En Linux, preferiblemente con [uv](https://docs.astral.sh/uv/) (descarga un
Python 3.12 que ya incluye Tk; sin uv usa `python3.12` con `venv` y `python3-tk`
del sistema):

```bash
./build_linux.sh
```

Genera `dist/Transcriptor` y `dist/Transcriptor-linux-<arquitectura>.tar.gz`. El
ejecutable funciona en distribuciones con una glibc igual o más nueva que la del
equipo donde se compila: para repartirlo, compila en una distribución antigua (la CI
usa Ubuntu 22.04).

Para ejecutar desde el código fuente:

```bat
pip install -r requirements.txt
python main.py
```

Autotest del ejecutable (lo usa la CI): `Transcriptor.exe --self-test [--model tiny] [--report informe.txt]`
(en Mac: `Transcriptor.app/Contents/MacOS/Transcriptor --self-test …`; en Linux:
`./Transcriptor --self-test …`, y con pantalla también abre una ventana Tk de prueba).

## Estructura

- `main.py` — punto de entrada.
- `transcriptor/gui.py` — interfaz (tkinter).
- `transcriptor/engine.py` — descarga/carga del modelo y transcripción en segundo plano.
- `transcriptor/audio.py` — decodificación en streaming y troceado por silencios (PyAV).
- `transcriptor/markdown.py` — escritura incremental del `.md`.
- `transcriptor/paths.py` — carpeta Descargas real de Windows, ajustes y registros.
- `transcriptor/power.py` — evita la limitación en segundo plano y el reposo.
- `Transcriptor.spec` — configuración de PyInstaller (un solo archivo en Windows y
  Linux, `.app` en macOS).

Los registros de errores están en `%LOCALAPPDATA%\Transcriptor\transcriptor.log`
(Windows), `~/Library/Application Support/Transcriptor/transcriptor.log` (Mac) o
`~/.local/share/Transcriptor/transcriptor.log` (Linux).
