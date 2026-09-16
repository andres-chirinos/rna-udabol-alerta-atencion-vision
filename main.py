import os
import sys
import time
import argparse
import yaml
import cv2
import numpy as np

from src.capture.camera_stream import CameraStream
from src.core.detector import FaceMeshDetector
from src.core.early_warning import EarlyWarningSystem, DriverAlertState
from src.alerts.alarm_manager import AlarmManager
from src.ui.hud import HUDDisplay
from src.decision.telemetry import TelemetryLogger


def load_config(config_path: str = "config/settings.yaml") -> dict:
    if not os.path.exists(config_path):
        print(f"[ADVERTENCIA] Archivo de configuración {config_path} no encontrado. Usando valores por defecto.")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Sistema de Visión Artificial para Detección Temprana de Microsueños")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Ruta al archivo de configuración YAML")
    parser.add_argument("--video", type=str, default=None, help="Ruta a un archivo de video para pruebas (opcional)")
    parser.add_argument("--no-gui", action="store_true", help="Ejecutar en modo sin ventana gráfica (headless)")
    args = parser.parse_args()

    config = load_config(args.config)

    print("================================================================")
    print("   SISTEMA DE ALERTA TEMPRANA DE MICROSUEÑOS PARA CONDUCTORES   ")
    print("   OpenCV + MediaPipe Face Mesh (478 Landmarks 3D)              ")
    print("================================================================")

    # 1. Inicialización de componentes
    print("[INFO] Cargando detector facial MediaPipe con soporte CLAHE...")
    detector = FaceMeshDetector(config)

    print("[INFO] Inicializando motor de alerta temprana (PERCLOS, Blinks, MAR, Head Pose)...")
    early_warning = EarlyWarningSystem(config)

    print("[INFO] Inicializando subsistema de alarmas acústicas asíncronas...")
    alarm_manager = AlarmManager(config)

    print("[INFO] Configurando interfaz HUD y telemetría...")
    hud = HUDDisplay(config)
    telemetry = TelemetryLogger(config)

    # 2. Inicialización de fuente de video
    video_source = args.video if args.video is not None else config.get("camera", {}).get("device_id", 0)
    width = config.get("camera", {}).get("width", 640)
    height = config.get("camera", {}).get("height", 480)

    print(f"[INFO] Abriendo fuente de captura: {video_source}...")
    stream = CameraStream(src=video_source, width=width, height=height)
    stream.start()

    if not stream.is_opened():
        print(f"[ERROR] No se pudo abrir la cámara o video '{video_source}'.")
        print("[INFO] Generando simulación sintética de verificación...")
        run_synthetic_demo(detector, early_warning, alarm_manager, hud, telemetry, config, args.no_gui)
        return

    print("\n[LISTO] Monitoreo iniciado con éxito.")
    print(" -> Presione 'c' en la ventana para recalibrar línea base.")
    print(" -> Presione 'q' o ESC para salir.\n")

    # 3. Bucle principal de procesamiento
    fps = 0.0
    prev_time = time.time()
    frame_count = 0
    fps_smoothing = 0.9

    headless = args.no_gui or ("DISPLAY" not in os.environ and sys.platform.startswith("linux"))

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                if args.video is not None:
                    print("[INFO] Fin del video de prueba.")
                    break
                time.sleep(0.01)
                continue

            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time
            if dt > 0:
                current_fps = 1.0 / dt
                fps = (fps_smoothing * fps) + ((1.0 - fps_smoothing) * current_fps)

            # Inferencia y extracción biométrica
            detection = detector.process_frame(frame)

            if detection["face_detected"]:
                ear = detection["ear"]
                mar = detection["mar"]
                pitch = detection["pitch"]
                yaw = detection["yaw"]

                # Actualización de FSM e índice EWFI
                state, score, diagnostics = early_warning.update(ear, mar, pitch, yaw)
            else:
                state = DriverAlertState.ALERT
                score = 0.0
                diagnostics = {"calibrating": False, "perclos": 0.0, "slow_blinks_count": 0, "last_blink_duration_ms": 0.0}

            # Emisión de alarmas sonoras no bloqueantes
            alarm_manager.trigger_alert(state, score)

            # Registro en log CSV
            telemetry.log_event(state, score, detection, diagnostics)

            # Renderizado del HUD
            annotated_frame = hud.render(frame, detection, state, score, diagnostics, fps)

            # Visualización interactiva
            if not headless:
                try:
                    cv2.imshow("Monitoreo de Microsueños - Driver Safety", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        break
                    elif key == ord('c'):
                        print("[INFO] Forzando recalibración de línea base...")
                        early_warning.is_calibrated = False
                        early_warning.calibration_ears.clear()
                        early_warning.start_time = time.time()
                except cv2.error:
                    headless = True

            frame_count += 1
            if headless and frame_count % 60 == 0:
                print(f"[TELEMETRIA] FPS: {fps:.1f} | Estado: {state.value} | EWFI: {score:.1f}/100 | EAR: {detection.get('ear', 0.0):.3f} | PERCLOS: {diagnostics.get('perclos', 0.0)*100:.1f}%")

    except KeyboardInterrupt:
        print("\n[INFO] Detención solicitada por el usuario.")

    finally:
        print("[INFO] Limpiando recursos...")
        stream.stop()
        alarm_manager.cleanup()
        if not headless:
            cv2.destroyAllWindows()
        print("[FIN] Sistema detenido correctamente.")


def run_synthetic_demo(detector, early_warning, alarm_manager, hud, telemetry, config, headless: bool):
    """Ejecuta una validación sintética para verificar que todos los subsistemas interactúan con cero errores."""
    print("[INFO] Ejecutando simulación de validación...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "MODO DE PRUEBA SINTETICO", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    detection = detector.process_frame(dummy_frame)
    state, score, diagnostics = early_warning.update(0.28, 0.20, 0.0, 0.0)
    annotated = hud.render(dummy_frame, detection, state, score, diagnostics, 30.0)
    print(f"[OK] Pipeline probado: Face Detected={detection['face_detected']}, Estado={state.value}")


if __name__ == "__main__":
    main()
