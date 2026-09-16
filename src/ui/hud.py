import cv2
import numpy as np
from typing import Dict, Any, Optional
from src.core.early_warning import DriverAlertState

# Paleta de colores BGR para interfaz HUD
COLOR_GREEN = (46, 204, 113)     # Alerta óptima
COLOR_YELLOW = (0, 215, 255)     # Atención temprana
COLOR_ORANGE = (39, 127, 243)    # Fatiga moderada
COLOR_RED = (0, 0, 255)          # Peligro crítico
COLOR_BLUE = (255, 150, 50)      # Calibrando
COLOR_MAGENTA = (255, 0, 255)    # Distracción
COLOR_BG_DARK = (20, 20, 20)     # Panel de fondo semitransparente
COLOR_WHITE = (245, 245, 245)
COLOR_GRAY = (140, 140, 140)


class HUDDisplay:
    """
    Renderizador Head-Up Display (HUD) con telemetría en tiempo real sobre OpenCV.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def render(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any],
        state: DriverAlertState,
        score: float,
        diagnostics: Dict[str, Any],
        fps: float
    ) -> np.ndarray:
        output_frame = frame.copy()
        h, w = output_frame.shape[:2]

        # 1. Dibujar contornos oculares si hay detección
        if detection.get("face_detected", False):
            self._draw_eye_landmarks(output_frame, detection, state)
            self._draw_head_pose_axis(output_frame, detection)

        # 2. Banner superior de estado
        self._draw_status_banner(output_frame, state, score, diagnostics, w)

        # 3. Panel lateral de telemetría y métricas
        self._draw_telemetry_panel(output_frame, detection, score, diagnostics, fps, h, w)

        return output_frame

    def _get_state_color(self, state: DriverAlertState) -> tuple:
        if state == DriverAlertState.ALERT:
            return COLOR_GREEN
        elif state == DriverAlertState.ATTENTION:
            return COLOR_YELLOW
        elif state == DriverAlertState.WARNING:
            return COLOR_ORANGE
        elif state == DriverAlertState.CRITICAL:
            return COLOR_RED
        elif state == DriverAlertState.DISTRACTED:
            return COLOR_MAGENTA
        elif state == DriverAlertState.CALIBRATING:
            return COLOR_BLUE
        return COLOR_GRAY

    def _draw_eye_landmarks(self, frame: np.ndarray, detection: Dict[str, Any], state: DriverAlertState):
        """Dibuja polígonos alrededor de los ojos con color según estado."""
        left_pts = detection.get("left_eye_pts")
        right_pts = detection.get("right_eye_pts")

        color = self._get_state_color(state)

        for eye_pts in [left_pts, right_pts]:
            if eye_pts is not None and len(eye_pts) >= 6:
                pts_int = eye_pts.astype(np.int32)
                cv2.polylines(frame, [pts_int], True, color, 1, cv2.LINE_AA)
                for pt in pts_int:
                    cv2.circle(frame, tuple(pt), 2, (0, 255, 255), -1, cv2.LINE_AA)

    def _draw_head_pose_axis(self, frame: np.ndarray, detection: Dict[str, Any]):
        """Proyecta un vector 3D desde la nariz mostrando la orientación de la cabeza."""
        rvec = detection.get("rvec")
        tvec = detection.get("tvec")
        landmarks = detection.get("landmarks")
        if rvec is None or tvec is None or landmarks is None or 1 not in landmarks:
            return

        h, w = frame.shape[:2]
        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Vector hacia el frente (eje Z azul de 100 mm)
        axis_3d = np.array([[0.0, 0.0, 120.0]], dtype=np.float64)
        imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)

        nose_pt = (int(landmarks[1][0] * w), int(landmarks[1][1] * h))
        target_pt = (int(imgpts[0][0][0]), int(imgpts[0][0][1]))

        cv2.arrowedLine(frame, nose_pt, target_pt, (255, 200, 0), 2, cv2.LINE_AA, tipLength=0.25)

    def _draw_status_banner(self, frame: np.ndarray, state: DriverAlertState, score: float, diagnostics: Dict[str, Any], width: int):
        """Banner superior con fondo semitransparente."""
        color = self._get_state_color(state)

        # Fondo
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 50), COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Barra indicadora de color
        cv2.rectangle(frame, (0, 0), (width, 6), color, -1)

        # Texto del estado
        state_text = f"ESTADO: {state.value}"
        if state == DriverAlertState.CALIBRATING:
            prog = diagnostics.get("progress", 0.0) * 100
            state_text += f" ({prog:.0f}%)"

        cv2.putText(frame, state_text, (15, 34), cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2, cv2.LINE_AA)

        # Advertencia de alarma crítica parpadeante
        if state == DriverAlertState.CRITICAL:
            cv2.putText(frame, "¡DESPIERTE! ALERTA MICROSUEÑO", (width - 420, 34),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, COLOR_RED, 2, cv2.LINE_AA)

    def _draw_telemetry_panel(self, frame: np.ndarray, detection: Dict[str, Any], score: float, diagnostics: Dict[str, Any], fps: float, height: int, width: int):
        """Panel lateral translúcido con todas las métricas biométricas."""
        panel_w = 260
        panel_h = 240
        x1 = width - panel_w - 10
        y1 = 60
        x2 = width - 10
        y2 = y1 + panel_h

        # Cuadro translúcido
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_BG_DARK, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 60, 60), 1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        # Título
        cv2.putText(frame, "TELEMETRIA DE FATIGA", (x1 + 10, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.line(frame, (x1 + 10, y1 + 28), (x2 - 10, y1 + 28), (80, 80, 80), 1)

        # Barra de puntuación EWFI (0 - 100)
        bar_x = x1 + 10
        bar_y = y1 + 42
        bar_w = panel_w - 20
        bar_h = 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int((min(100.0, score) / 100.0) * bar_w)
        score_color = COLOR_GREEN if score < 35 else (COLOR_YELLOW if score < 60 else COLOR_RED)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), score_color, -1)

        cv2.putText(frame, f"Indice Fatiga (EWFI): {score:.1f}/100", (x1 + 10, y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1, cv2.LINE_AA)

        # Métricas individuales
        ear = detection.get("ear", 0.0)
        mar = detection.get("mar", 0.0)
        pitch = detection.get("pitch", 0.0)
        yaw = detection.get("yaw", 0.0)
        perclos = diagnostics.get("perclos", 0.0) * 100
        slow_blinks = diagnostics.get("slow_blinks_count", 0)
        last_blink_ms = diagnostics.get("last_blink_duration_ms", 0.0)
        yawns = diagnostics.get("yawns_count", 0)

        lines = [
            (f"EAR (Apertura Ojos): {ear:.3f}", COLOR_WHITE),
            (f"PERCLOS (Ventana 15s): {perclos:.1f}%", COLOR_YELLOW if perclos > 15 else COLOR_WHITE),
            (f"Ultimo Parpadeo: {last_blink_ms:.0f} ms", COLOR_RED if last_blink_ms > 380 else COLOR_WHITE),
            (f"Parpadeos Lentos/min: {slow_blinks}", COLOR_RED if slow_blinks >= 2 else COLOR_WHITE),
            (f"MAR (Bostezo): {mar:.2f} (Tot: {yawns})", COLOR_YELLOW if mar > 0.60 else COLOR_WHITE),
            (f"Inclinacion Cefalica: {pitch:.1f} deg", COLOR_RED if pitch < -16 else COLOR_WHITE),
            (f"FPS Inferencia: {fps:.1f}", COLOR_GREEN)
        ]

        text_y = y1 + 72
        for line_text, text_color in lines:
            cv2.putText(frame, line_text, (x1 + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_color, 1, cv2.LINE_AA)
            text_y += 22
