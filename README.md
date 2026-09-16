# Sistema de Visión Artificial para la Detección Temprana de Microsueños y Fatiga en Conductores

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-1.0.1-orange.svg)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sistema biométrico de visión artificial en tiempo real diseñado para prevenir siniestros viales mediante la identificación prodrómica y temprana de signos de fatiga y microsueños en conductores de vehículos y operadores de maquinaria.

A diferencia de las soluciones reactivas convencionales que solo disparan alarmas cuando el conductor ya lleva más de un segundo con los párpados cerrados, este sistema incorpora un **Índice Compuesto de Fatiga Temprana (EWFI)** que analiza la microdinámica del parpadeo lento, la tasa de apertura bucal (bostezos), el porcentaje de cierre palpebral (PERCLOS) y la estabilidad postural tridimensional de la cabeza antes de que ocurra la pérdida total de la consciencia.

---

## Características Principales

* **Detección Ocular de Alta Fidelidad (EAR & PERCLOS):**
  * Cálculo continuo del *Eye Aspect Ratio* (EAR) vectorial sobre los 6 puntos clave de cada ojo.
  * Medición de PERCLOS sobre una ventana móvil configurable de 15 segundos.
  * Cronómetro de parpadeos individuales con resolución de milisegundos para detectar **parpadeos lentos patológicos** ($\ge 380\text{ ms}$).
* **Detección de Bostezos (MAR):**
  * Cálculo del *Mouth Aspect Ratio* (MAR) tridimensional normalizado frente a distancias anatómicas labiales.
  * Acumulación temporal de bostezos confirmados en ventanas de 60 segundos.
* **Estimación de Pose de Cabeza 3D (Head Pose via `cv2.solvePnP`):**
  * Resolución geométrica de perspectiva respecto a un modelo antropométrico canónico.
  * Detección de cabeceo involuntario (*nodding/pitch drop*) y distracción lateral de la vía (*yaw deviation*).
* **Calibración Adaptativa Dinámica:**
  * Auto-calibración de la línea base ($EAR_{base}$) durante los primeros 3 segundos de operación por conductor, eliminando falsos positivos por morfología ocular o ángulo de la cámara.
* **Preprocesamiento Óptico Adaptativo (CLAHE):**
  * Ecualización en el canal de luminancia (espacio LAB) para mantener alta sensibilidad ante sombras en la cabina o deslumbramientos diurnos.
* **Alarmas Acústicas Asíncronas No Bloqueantes:**
  * Sintetizador de tonos de audio PCM en memoria mediante Pygame (sin requerir ficheros WAV externos).
  * Niveles escalonados: Tono suave de atención (880 Hz), tono preventivo doble (650 Hz) y sirena crítica pulsante (1200 Hz).
* **Interfaz Gráfica HUD (Head-Up Display):**
  * Superposición en pantalla con OpenCV mostrando contornos oculares, vector 3D de mirada, banner cromático de alerta y tarjeta lateral de telemetría.
* **Auditoría y Telemetría:**
  * Registro automático de eventos e incidentes con marca de tiempo precisa en `data/logs/fatigue_events.csv`.

---

## Estructura del Proyecto

```text
redes-neuronales-proyecto/
├── config/
│   └── settings.yaml          # Umbrales, pesos de fatiga, parámetros de cámara
├── data/
│   ├── logs/                  # Registro de telemetría (fatigue_events.csv)
│   └── sounds/                # Archivos de audio auxiliares
├── docs/
│   └── ARQUITECTURA_Y_TECNOLOGIAS.md # Especificación técnica y marco teórico
├── models/
│   └── face_landmarker.task   # Modelo TFLite de MediaPipe FaceLandmarker
├── src/
│   ├── capture/
│   │   └── camera_stream.py   # Captura asíncrona multihilo en OpenCV
│   ├── core/
│   │   ├── detector.py        # Wrapper MediaPipe + preprocesamiento CLAHE
│   │   ├── early_warning.py   # Motor FSM y cálculo del índice EWFI
│   │   └── metrics.py         # Fórmulas algebraicas EAR, MAR y solvePnP
│   ├── alerts/
│   │   └── alarm_manager.py   # Sintetizador y emisor de alarmas multihilo
│   ├── decision/
│   │   └── telemetry.py       # Logger de auditoría CSV
│   └── ui/
│       └── hud.py             # Renderizador HUD en OpenCV
├── tests/
│   └── test_pipeline.py       # Pruebas unitarias de algoritmos y métricas
├── main.py                    # Punto de entrada de la aplicación
├── requirements.txt           # Dependencias de Python
└── README.md                  # Este documento
```

---

## Requisitos e Instalación

### Prerrequisitos
* Sistema Operativo: Linux (Ubuntu, Debian, Fedora, etc.), macOS o Windows.
* Python: versión 3.10 o superior (validado en Python 3.13).
* Cámara web USB o stream de video IP (RTSP).

### Instalación en Entorno Virtual

1. **Clonar o ingresar al repositorio:**
   ```bash
   cd redes-neuronales-proyecto
   ```

2. **Crear el entorno virtual:**
   ```bash
   python3 -m venv .venv
   ```

3. **Activar el entorno virtual:**
   * En Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```
   * En Windows:
     ```bash
     .venv\Scripts\activate
     ```

4. **Instalar dependencias:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Verificar el modelo de MediaPipe:**
   El modelo `models/face_landmarker.task` ya se encuentra descargado en el repositorio. Si necesitas regenerarlo:
   ```bash
   mkdir -p models
   curl -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
   ```

---

## Guía de Uso

### 1. Ejecución con Cámara Web en Vivo
Para iniciar el monitoreo en tiempo real con la cámara predeterminada (ID 0):
```bash
.venv/bin/python main.py
```

### 2. Ejecución sobre un Archivo de Video
Para validar o evaluar el sistema con grabaciones de pruebas:
```bash
.venv/bin/python main.py --video data/test_driver.mp4
```

### 3. Modo Headless (Servidores / Sin Pantalla Gráfica)
Para entornos embebidos o servidores sin servidor X11/Wayland:
```bash
.venv/bin/python main.py --no-gui
```

### Controles Interactivos en Ventana
| Tecla | Acción |
| :--- | :--- |
| `c` | Forzar **recalibración** de la línea base ocular ($EAR_{base}$) |
| `q` / `ESC` | Finalizar el programa y liberar recursos |

---

## Estados del Sistema y Niveles de Alarma

| Estado | EWFI Score | Color HUD | Alarma Sonora | Significado / Criterio |
| :--- | :--- | :--- | :--- | :--- |
| **`CALIBRANDO`** | — | Azul | Silencio | Primeros 3s registrando la apertura ocular natural del conductor. |
| **`ALERTA_OPTIMA`** | 0 – 34 | Verde | Silencio | Conductor en vigilia plena, parpadeos rápidos normales. |
| **`ATENCION_TEMPRANA`** | 35 – 59 | Amarillo | Tono suave (880 Hz) | Párpados pesados, parpadeo lento ($\ge 380\text{ ms}$) o bostezo. |
| **`FATIGA_MODERADA`** | 60 – 74 | Naranja | Tono doble (650 Hz) | PERCLOS $\ge 15\%$ o acumulación de bostezos en el último minuto. |
| **`MICROSUEÑO_CRITICO`** | $\ge 75$ | Rojo | Sirena continua (1200 Hz) | Cierre ocular $\ge 1.0\text{ s}$ o cabeceo con ojos cerrados. |
| **`DISTRACCION_RUTA`** | — | Magenta | Beep de atención | Desvío lateral de la cabeza ($|Yaw| > 28^\circ$) por más de 2 segundos. |

---

## Ejecución de Pruebas Unitarias

Para comprobar la integridad matemática de los algoritmos (EAR, MAR, solvePnP, lógica temporal de FSM y renderizado):
```bash
.venv/bin/python -m unittest tests/test_pipeline.py
```

---

## Documentación Técnica Adicional

Para consultar la fundamentación teórica, deducciones matemáticas detalladas y diagramas de arquitectura para la redacción de informes o tesis académica, revisa:
* [ARQUITECTURA_Y_TECNOLOGIAS.md](docs/ARQUITECTURA_Y_TECNOLOGIAS.md)
* [VISION ARTIFICIAL (rn).md](VISION%20ARTIFICIAL%20%28rn%29.md)
