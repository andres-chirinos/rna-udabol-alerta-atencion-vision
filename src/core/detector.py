import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

from src.core.metrics import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    HEAD_POSE_LANDMARK_IDS,
    MOUTH_OUTER_HORIZONTAL,
    MOUTH_VERTICAL_PAIRS,
    calculate_ear,
    calculate_mar,
    estimate_head_pose
)


class FaceMeshDetector:
    """
    Detector facial de alta precisión basado en MediaPipe FaceLandmarker y OpenCV.
    Aplica CLAHE adaptativo para resistir cambios de luz diurna/nocturna.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        model_path = config.get("detection", {}).get("model_path", "models/face_landmarker.task")

        # Configuración de MediaPipe FaceLandmarker
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=True,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=config.get("detection", {}).get("min_face_detection_confidence", 0.5),
            min_face_presence_confidence=config.get("detection", {}).get("min_face_presence_confidence", 0.5),
            min_tracking_confidence=config.get("detection", {}).get("min_tracking_confidence", 0.5)
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # Preprocesamiento CLAHE (Contrast Limited Adaptive Histogram Equalization)
        prep_cfg = config.get("preprocessing", {})
        self.enable_clahe = prep_cfg.get("enable_clahe", True)
        clip_limit = prep_cfg.get("clahe_clip_limit", 2.0)
        grid_size = tuple(prep_cfg.get("clahe_tile_grid_size", [8, 8]))
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    def preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Acondiciona la imagen en espacio LAB ecualizando luminosidad sin distorsionar color."""
        if not self.enable_clahe:
            return frame_bgr

        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_clahe = self.clahe.apply(l_chan)
        lab_enhanced = cv2.merge((l_clahe, a_chan, b_chan))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Procesa el frame BGR y extrae todas las métricas biométricas esenciales.
        """
        h, w = frame_bgr.shape[:2]
        enhanced_bgr = self.preprocess_frame(frame_bgr)
        rgb_frame = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.face_landmarks:
            return {
                "face_detected": False,
                "ear": 0.0,
                "left_ear": 0.0,
                "right_ear": 0.0,
                "mar": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "blendshapes": {},
                "landmarks": None,
                "rvec": None,
                "tvec": None
            }

        face_lms = detection_result.face_landmarks[0]

        # Mapeo de coordenadas normalizadas (0.0 a 1.0) a diccionario de puntos (x, y, z)
        lm_dict = {}
        for idx, lm in enumerate(face_lms):
            lm_dict[idx] = np.array([lm.x, lm.y, lm.z], dtype=np.float64)

        # 1. Puntos de ojos y cálculo de EAR
        left_eye_pts = np.array([[lm_dict[idx][0] * w, lm_dict[idx][1] * h] for idx in LEFT_EYE_INDICES])
        right_eye_pts = np.array([[lm_dict[idx][0] * w, lm_dict[idx][1] * h] for idx in RIGHT_EYE_INDICES])

        left_ear = calculate_ear(left_eye_pts)
        right_ear = calculate_ear(right_eye_pts)
        avg_ear = (left_ear + right_ear) / 2.0

        # 2. Puntos de boca y cálculo de MAR
        mouth_pts_2d = {}
        # Recolectar puntos de boca relevantes
        needed_mouth_ids = [MOUTH_OUTER_HORIZONTAL[0], MOUTH_OUTER_HORIZONTAL[1]]
        for pair in MOUTH_VERTICAL_PAIRS:
            needed_mouth_ids.extend(pair)

        for mid in set(needed_mouth_ids):
            if mid in lm_dict:
                mouth_pts_2d[mid] = np.array([lm_dict[mid][0] * w, lm_dict[mid][1] * h])

        mar = calculate_mar(mouth_pts_2d)

        # 3. Estimación de Head Pose (solvePnP)
        pitch, yaw, roll, rvec, tvec = estimate_head_pose(lm_dict, (h, w))

        # 4. Blendshapes neuronales de MediaPipe (parpadeos y apertura)
        blendshapes_dict = {}
        if detection_result.face_blendshapes:
            for category in detection_result.face_blendshapes[0]:
                blendshapes_dict[category.category_name] = category.score

        return {
            "face_detected": True,
            "ear": avg_ear,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "mar": mar,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "blendshapes": blendshapes_dict,
            "landmarks": lm_dict,
            "rvec": rvec,
            "tvec": tvec,
            "left_eye_pts": left_eye_pts,
            "right_eye_pts": right_eye_pts
        }
