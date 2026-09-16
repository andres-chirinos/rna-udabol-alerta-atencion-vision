import os
import csv
import time
from typing import Dict, Any
from src.core.early_warning import DriverAlertState


class TelemetryLogger:
    """
    Registrador de eventos biométricos e incidentes de fatiga para auditoría y análisis estadístico.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("alerts", {}).get("log_events_to_csv", True)
        self.csv_path = config.get("alerts", {}).get("csv_log_path", "data/logs/fatigue_events.csv")
        self.last_log_time = 0.0

        if self.enabled:
            try:
                os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
                if not os.path.exists(self.csv_path):
                    with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            "timestamp",
                            "iso_time",
                            "state",
                            "ewfi_score",
                            "ear",
                            "perclos_pct",
                            "last_blink_ms",
                            "slow_blinks_min",
                            "mar",
                            "yawns_count",
                            "pitch_deg",
                            "yaw_deg"
                        ])
            except Exception as e:
                print(f"[ADVERTENCIA] No se pudo crear archivo de log CSV: {e}. Telemetría en archivo desactivada.")
                self.enabled = False

    def log_event(self, state: DriverAlertState, score: float, detection: Dict[str, Any], diagnostics: Dict[str, Any]):
        """Registra un evento relevante (cambio de estado o intervalos periódicos en alerta)."""
        if not self.enabled:
            return

        now = time.time()
        # Registrar eventos anómalos o cada 5 segundos si el estado es normal
        is_anomaly = state in (
            DriverAlertState.ATTENTION,
            DriverAlertState.WARNING,
            DriverAlertState.CRITICAL,
            DriverAlertState.DISTRACTED
        )
        if not is_anomaly and (now - self.last_log_time < 5.0):
            return

        self.last_log_time = now
        iso_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))

        try:
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    round(now, 3),
                    iso_time,
                    state.value,
                    round(score, 1),
                    round(detection.get("ear", 0.0), 3),
                    round(diagnostics.get("perclos", 0.0) * 100, 1),
                    round(diagnostics.get("last_blink_duration_ms", 0.0), 1),
                    diagnostics.get("slow_blinks_count", 0),
                    round(detection.get("mar", 0.0), 3),
                    diagnostics.get("yawns_count", 0),
                    round(detection.get("pitch", 0.0), 1),
                    round(detection.get("yaw", 0.0), 1)
                ])
        except Exception:
            pass
