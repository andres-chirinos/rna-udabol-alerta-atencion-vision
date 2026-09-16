import time
from collections import deque
from typing import Dict, Any, Tuple, Optional
from enum import Enum


class DriverAlertState(Enum):
    CALIBRATING = "CALIBRANDO"
    ALERT = "ALERTA_OPTIMA"
    ATTENTION = "ATENCION_TEMPRANA"
    WARNING = "FATIGA_MODERADA"
    CRITICAL = "MICROSUEÑO_CRITICO"
    DISTRACTED = "DISTRACCION_RUTA"


class EarlyWarningSystem:
    """
    Sistema integral de alerta temprana para microsueños y fatiga.
    Analiza PERCLOS en ventana móvil, duración exacta de parpadeos individuales,
    bostezos acumulados y estabilidad cefálica.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = config.get("thresholds", {})
        self.weights = config.get("early_warning_weights", {
            "perclos": 0.35,
            "slow_blinks": 0.25,
            "yawns": 0.20,
            "head_instability": 0.20
        })
        self.alert_levels = config.get("alert_levels", {
            "attention_score": 35.0,
            "warning_score": 60.0,
            "critical_score": 75.0
        })

        # Calibración adaptativa
        self.calibration_duration = config.get("system", {}).get("calibration_duration_sec", 3.0)
        self.start_time = time.time()
        self.is_calibrated = False
        self.calibration_ears = []
        self.baseline_ear = self.thresholds.get("ear_absolute_closed", 0.21) / 0.75

        # Umbrales calculados
        self.ear_closed_thresh = self.thresholds.get("ear_absolute_closed", 0.21)
        self.ear_drowsy_thresh = self.thresholds.get("ear_drowsy_closed", 0.24)
        self.slow_blink_ms = self.thresholds.get("blink_slow_duration_ms", 380)
        self.microsleep_ms = self.thresholds.get("microsleep_duration_ms", 1000)

        # Buffer para PERCLOS (almacena tuples: (timestamp, is_closed_or_drooping))
        self.perclos_window_sec = self.thresholds.get("perclos_window_seconds", 15.0)
        self.ear_history = deque()

        # Rastreo de parpadeos individuales
        self.eye_closed_start_time: Optional[float] = None
        self.current_closure_duration_ms: float = 0.0
        self.recent_blinks = deque()  # (timestamp, duration_ms, is_slow)
        self.last_blink_duration_ms: float = 0.0

        # Bostezos
        self.mar_thresh = self.thresholds.get("mar_yawn_threshold", 0.62)
        self.yawn_min_sec = self.thresholds.get("yawn_min_duration_sec", 1.5)
        self.yawn_start_time: Optional[float] = None
        self.recent_yawns = deque()  # timestamps de bostezos confirmados

        # Cabeceo / Distracción
        self.nod_pitch_thresh = self.thresholds.get("nod_pitch_threshold", -16.0)
        self.nod_start_time: Optional[float] = None
        self.nod_duration_sec: float = 0.0
        self.recent_nods = deque()

        self.distraction_yaw_thresh = self.thresholds.get("distraction_yaw_threshold", 28.0)
        self.distraction_start_time: Optional[float] = None
        self.distraction_duration_sec: float = 0.0

        # Estado actual
        self.current_state = DriverAlertState.CALIBRATING
        self.current_score = 0.0

    def update(self, ear: float, mar: float, pitch: float, yaw: float) -> Tuple[DriverAlertState, float, Dict[str, Any]]:
        """
        Actualiza el estado temporal con las métricas del frame actual.
        Retorna: (Estado, Puntuación EWFI 0-100, Diccionario de diagnósticos detallados)
        """
        now = time.time()

        # 1. Fase de calibración inicial
        if not self.is_calibrated:
            if now - self.start_time < self.calibration_duration:
                if ear > 0.15:  # Evita parpadeos esporádicos durante calibración
                    self.calibration_ears.append(ear)
                return DriverAlertState.CALIBRATING, 0.0, {
                    "calibrating": True,
                    "progress": (now - self.start_time) / self.calibration_duration
                }
            else:
                if len(self.calibration_ears) > 10:
                    self.baseline_ear = float(sum(self.calibration_ears) / len(self.calibration_ears))
                    multiplier = self.thresholds.get("ear_baseline_multiplier", 0.75)
                    self.ear_closed_thresh = max(0.18, self.baseline_ear * multiplier)
                    self.ear_drowsy_thresh = max(0.21, self.baseline_ear * 0.85)
                self.is_calibrated = True

        # 2. Análisis de parpadeo y cierre palpebral continuo
        is_closed = ear < self.ear_closed_thresh
        is_drooping = ear < self.ear_drowsy_thresh

        # Registro para PERCLOS
        self.ear_history.append((now, is_closed or is_drooping))
        cutoff_perclos = now - self.perclos_window_sec
        while self.ear_history and self.ear_history[0][0] < cutoff_perclos:
            self.ear_history.popleft()

        closed_count = sum(1 for _, closed in self.ear_history if closed)
        perclos_val = closed_count / max(1, len(self.ear_history))

        # Seguimiento de duración de cierre
        if is_closed:
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = now
            self.current_closure_duration_ms = (now - self.eye_closed_start_time) * 1000.0
        else:
            if self.eye_closed_start_time is not None:
                duration_ms = (now - self.eye_closed_start_time) * 1000.0
                self.last_blink_duration_ms = duration_ms
                is_slow = duration_ms >= self.slow_blink_ms
                self.recent_blinks.append((now, duration_ms, is_slow))
                self.eye_closed_start_time = None
                self.current_closure_duration_ms = 0.0

        # Limpiar historial de parpadeos de más de 60 segundos
        cutoff_blinks = now - 60.0
        while self.recent_blinks and self.recent_blinks[0][0] < cutoff_blinks:
            self.recent_blinks.popleft()

        slow_blinks_count = sum(1 for _, _, is_slow in self.recent_blinks if is_slow)
        total_blinks_min = len(self.recent_blinks)

        # 3. Detección de bostezos (MAR)
        is_yawning = mar > self.mar_thresh
        if is_yawning:
            if self.yawn_start_time is None:
                self.yawn_start_time = now
            elif now - self.yawn_start_time >= self.yawn_min_sec:
                # Bostezo confirmado
                if not self.recent_yawns or (now - self.recent_yawns[-1] > 4.0):
                    self.recent_yawns.append(now)
        else:
            self.yawn_start_time = None

        cutoff_yawns = now - self.thresholds.get("yawn_history_window_sec", 60.0)
        while self.recent_yawns and self.recent_yawns[0] < cutoff_yawns:
            self.recent_yawns.popleft()
        yawns_count = len(self.recent_yawns)

        # 4. Orientación de Cabeza: Cabeceo (Nodding) y Distracción
        if pitch < self.nod_pitch_thresh:
            if self.nod_start_time is None:
                self.nod_start_time = now
            self.nod_duration_sec = now - self.nod_start_time
        else:
            if self.nod_start_time is not None:
                if now - self.nod_start_time >= self.thresholds.get("nod_min_duration_sec", 1.2):
                    self.recent_nods.append(now)
                self.nod_start_time = None
                self.nod_duration_sec = 0.0

        cutoff_nods = now - 60.0
        while self.recent_nods and self.recent_nods[0] < cutoff_nods:
            self.recent_nods.popleft()
        nods_count = len(self.recent_nods)

        # Distracción lateral
        if abs(yaw) > self.distraction_yaw_thresh:
            if self.distraction_start_time is None:
                self.distraction_start_time = now
            self.distraction_duration_sec = now - self.distraction_start_time
        else:
            self.distraction_start_time = None
            self.distraction_duration_sec = 0.0

        # 5. Cálculo del Índice de Fatiga Temprana (EWFI: 0 a 100)
        # Componente PERCLOS: normalizado contra 0.25 (25% = 100 puntos)
        score_perclos = min(100.0, (perclos_val / 0.25) * 100.0)

        # Componente Parpadeos Lentos: 3 parpadeos lentos/minuto = 100 puntos
        score_blinks = min(100.0, (slow_blinks_count / 3.0) * 100.0)

        # Componente Bostezos: 2 bostezos/minuto = 100 puntos
        score_yawns = min(100.0, (yawns_count / 2.0) * 100.0)

        # Componente Inestabilidad Cefálica: cabeceos
        score_head = min(100.0, ((nods_count * 50.0) + (min(1.0, self.nod_duration_sec / 1.5) * 50.0)))

        fused_score = (
            score_perclos * self.weights["perclos"] +
            score_blinks * self.weights["slow_blinks"] +
            score_yawns * self.weights["yawns"] +
            score_head * self.weights["head_instability"]
        )
        self.current_score = fused_score

        # 6. Determinación de Estado
        # Condición inmediata de emergencia (Microsueño activo)
        is_immediate_microsleep = (
            self.current_closure_duration_ms >= self.microsleep_ms or
            (self.nod_duration_sec >= 1.5 and is_closed)
        )

        is_distracted = self.distraction_duration_sec >= self.thresholds.get("distraction_duration_sec", 2.0)

        has_sufficient_perclos_samples = len(self.ear_history) >= 20
        perclos_critical_triggered = has_sufficient_perclos_samples and (perclos_val >= self.thresholds.get("perclos_critical", 0.25))
        perclos_warning_triggered = has_sufficient_perclos_samples and (perclos_val >= self.thresholds.get("perclos_warning", 0.15))

        if is_immediate_microsleep:
            state = DriverAlertState.CRITICAL
        elif is_distracted:
            state = DriverAlertState.DISTRACTED
        elif fused_score >= self.alert_levels["critical_score"] or perclos_critical_triggered:
            state = DriverAlertState.CRITICAL
        elif fused_score >= self.alert_levels["warning_score"] or perclos_warning_triggered:
            state = DriverAlertState.WARNING
        elif fused_score >= self.alert_levels["attention_score"] or slow_blinks_count >= 1 or yawns_count >= 1:
            state = DriverAlertState.ATTENTION
        else:
            state = DriverAlertState.ALERT

        self.current_state = state

        diagnostics = {
            "calibrating": False,
            "baseline_ear": self.baseline_ear,
            "ear_closed_threshold": self.ear_closed_thresh,
            "perclos": perclos_val,
            "current_closure_duration_ms": self.current_closure_duration_ms,
            "last_blink_duration_ms": self.last_blink_duration_ms,
            "slow_blinks_count": slow_blinks_count,
            "total_blinks_min": total_blinks_min,
            "yawns_count": yawns_count,
            "nods_count": nods_count,
            "nod_duration_sec": self.nod_duration_sec,
            "distraction_duration_sec": self.distraction_duration_sec,
            "ewfi_score": fused_score,
            "sub_scores": {
                "perclos": score_perclos,
                "slow_blinks": score_blinks,
                "yawns": score_yawns,
                "head": score_head
            }
        }

        return state, fused_score, diagnostics
