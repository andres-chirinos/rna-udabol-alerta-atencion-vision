import unittest
import numpy as np
import yaml
import time

from src.core.metrics import (
    calculate_ear,
    calculate_mar,
    estimate_head_pose,
    LEFT_EYE_INDICES,
    HEAD_POSE_LANDMARK_IDS
)
from src.core.early_warning import EarlyWarningSystem, DriverAlertState
from src.ui.hud import HUDDisplay


class TestEarlyMicrosleepDetection(unittest.TestCase):

    def setUp(self):
        with open("config/settings.yaml", "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def test_ear_calculation(self):
        """Verifica que el cálculo de EAR diferencie ojos abiertos de ojos cerrados."""
        # Coordenadas simuladas de ojo abierto (distancia vertical amplia)
        open_eye = np.array([
            [10.0, 50.0],  # p1
            [20.0, 40.0],  # p2
            [30.0, 40.0],  # p3
            [40.0, 50.0],  # p4
            [30.0, 60.0],  # p5
            [20.0, 60.0],  # p6
        ])
        ear_open = calculate_ear(open_eye)
        self.assertGreater(ear_open, 0.30, "El EAR de ojo abierto debe ser superior a 0.30")

        # Coordenadas simuladas de ojo cerrado (párpados casi pegados)
        closed_eye = np.array([
            [10.0, 50.0],
            [20.0, 49.0],
            [30.0, 49.0],
            [40.0, 50.0],
            [30.0, 51.0],
            [20.0, 51.0],
        ])
        ear_closed = calculate_ear(closed_eye)
        self.assertLess(ear_closed, 0.15, "El EAR de ojo cerrado debe ser inferior a 0.15")

    def test_mar_calculation(self):
        """Verifica cálculo de MAR para boca cerrada vs bostezo abierto."""
        mouth_closed = {
            61: np.array([100.0, 200.0]),
            291: np.array([200.0, 200.0]),
            13: np.array([150.0, 198.0]),
            14: np.array([150.0, 202.0]),
            81: np.array([125.0, 198.0]),
            178: np.array([125.0, 202.0]),
            311: np.array([175.0, 198.0]),
            402: np.array([175.0, 202.0]),
        }
        mar_closed = calculate_mar(mouth_closed)
        self.assertLess(mar_closed, 0.20, "MAR de boca cerrada debe ser bajo (<0.20)")

        mouth_yawn = {
            61: np.array([100.0, 200.0]),
            291: np.array([200.0, 200.0]),
            13: np.array([150.0, 160.0]),
            14: np.array([150.0, 240.0]),
            81: np.array([125.0, 170.0]),
            178: np.array([125.0, 230.0]),
            311: np.array([175.0, 170.0]),
            402: np.array([175.0, 230.0]),
        }
        mar_yawn = calculate_mar(mouth_yawn)
        self.assertGreater(mar_yawn, 0.65, "MAR de bostezo debe ser elevado (>0.65)")

    def test_early_warning_state_progression(self):
        """Verifica la transición de estados de la FSM de fatiga."""
        # Forzar calibración rápida para pruebas
        self.config["system"]["calibration_duration_sec"] = 0.1
        ews = EarlyWarningSystem(self.config)

        # Simular calibración con EAR normal (0.30)
        time.sleep(0.15)
        state, score, _ = ews.update(0.30, 0.15, 0.0, 0.0)
        self.assertIn(state, [DriverAlertState.ALERT, DriverAlertState.CALIBRATING])

        # Forzar calibrado
        ews.is_calibrated = True
        ews.baseline_ear = 0.30
        ews.ear_closed_thresh = 0.21

        # Simular estado óptimo
        state, score, diag = ews.update(0.30, 0.15, 0.0, 0.0)
        self.assertEqual(state, DriverAlertState.ALERT)
        self.assertLess(score, 35.0)

        # Simular parpadeo patológico prolongado (ojo cerrado > slow_blink_ms)
        ews.eye_closed_start_time = time.time() - 0.50  # 500 ms cerrado
        ews.update(0.12, 0.15, 0.0, 0.0)
        # Reabrir ojo -> registra parpadeo lento
        state, score, diag = ews.update(0.30, 0.15, 0.0, 0.0)
        self.assertGreaterEqual(diag["slow_blinks_count"], 1)
        self.assertIn(state, [DriverAlertState.ATTENTION, DriverAlertState.WARNING])

        # Simular cierre sostenido crítico (> 1000 ms = microsueño inmediato)
        ews.eye_closed_start_time = time.time() - 1.20  # 1.2 segundos
        state, score, diag = ews.update(0.10, 0.15, 0.0, 0.0)
        self.assertEqual(state, DriverAlertState.CRITICAL)

    def test_hud_rendering(self):
        """Verifica que el renderizador de HUD procese frames sin fallos ni desbordamientos."""
        hud = HUDDisplay(self.config)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detection = {
            "face_detected": True,
            "ear": 0.28,
            "mar": 0.18,
            "pitch": -5.0,
            "yaw": 2.0,
            "roll": 0.0,
            "left_eye_pts": np.array([[100, 100], [110, 95], [120, 95], [130, 100], [120, 105], [110, 105]]),
            "right_eye_pts": np.array([[200, 100], [210, 95], [220, 95], [230, 100], [220, 105], [210, 105]]),
            "rvec": np.zeros((3, 1)),
            "tvec": np.array([[0.0], [0.0], [500.0]]),
            "landmarks": {1: np.array([0.5, 0.5, 0.0])}
        }
        diagnostics = {
            "calibrating": False,
            "perclos": 0.08,
            "last_blink_duration_ms": 220.0,
            "slow_blinks_count": 0,
            "yawns_count": 0
        }
        output = hud.render(dummy_frame, detection, DriverAlertState.ALERT, 15.0, diagnostics, 30.0)
        self.assertEqual(output.shape, (480, 640, 3))
        self.assertGreater(output.sum(), 0, "El frame anotado debe contener píxeles dibujados")


if __name__ == "__main__":
    unittest.main()
