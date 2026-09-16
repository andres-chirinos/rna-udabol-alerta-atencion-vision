import cv2
import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional

# Índices MediaPipe Face Mesh para ojos y boca
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Puntos de boca para MAR
MOUTH_OUTER_HORIZONTAL = (61, 291)
MOUTH_VERTICAL_PAIRS = [
    (13, 14),   # Centro labio superior/inferior
    (81, 178),  # Izquierda
    (311, 402)  # Derecha
]

# Puntos para estimación 3D Head Pose
# Modelo canónico 3D de rostro humano en mm (antropométrico estándar)
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],          # Punta de la nariz (Landmark 1)
    [0.0, -330.0, -65.0],     # Mentón (Landmark 152)
    [-225.0, 170.0, -135.0],  # Esquina exterior ojo izq (Landmark 33)
    [225.0, 170.0, -135.0],   # Esquina exterior ojo der (Landmark 263)
    [-150.0, -150.0, -125.0], # Comisura labial izq (Landmark 61)
    [150.0, -150.0, -125.0]   # Comisura labial der (Landmark 291)
], dtype=np.float64)

HEAD_POSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def compute_euclidean_distance(pt1: np.ndarray, pt2: np.ndarray) -> float:
    """Calcula distancia euclidiana 2D o 3D entre dos puntos."""
    return float(np.linalg.norm(pt1 - pt2))


def calculate_ear(eye_points: np.ndarray) -> float:
    """
    Calcula el Eye Aspect Ratio (EAR) según Soukupová & Čech (2016).
    eye_points debe contener 6 puntos en orden: [p1, p2, p3, p4, p5, p6]
    p1, p4: esquinas horizontales
    p2, p6 y p3, p5: pares verticales
    """
    if len(eye_points) < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_points[:6]

    v1 = compute_euclidean_distance(p2, p6)
    v2 = compute_euclidean_distance(p3, p5)
    horizontal = compute_euclidean_distance(p1, p4)

    if horizontal < 1e-6:
        return 0.0

    return (v1 + v2) / (2.0 * horizontal)


def calculate_mar(mouth_points_dict: Dict[int, np.ndarray]) -> float:
    """
    Calcula el Mouth Aspect Ratio (MAR) para detección de bostezo.
    """
    h_start_id, h_end_id = MOUTH_OUTER_HORIZONTAL
    if h_start_id not in mouth_points_dict or h_end_id not in mouth_points_dict:
        return 0.0

    h_dist = compute_euclidean_distance(mouth_points_dict[h_start_id], mouth_points_dict[h_end_id])
    if h_dist < 1e-6:
        return 0.0

    vertical_sum = 0.0
    valid_pairs = 0
    for upper_id, lower_id in MOUTH_VERTICAL_PAIRS:
        if upper_id in mouth_points_dict and lower_id in mouth_points_dict:
            vertical_sum += compute_euclidean_distance(mouth_points_dict[upper_id], mouth_points_dict[lower_id])
            valid_pairs += 1

    if valid_pairs == 0:
        return 0.0

    return vertical_sum / (valid_pairs * h_dist)


def estimate_head_pose(landmarks_dict: Dict[int, np.ndarray], image_shape: Tuple[int, int]) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Estima los ángulos de Euler de la cabeza (Pitch, Yaw, Roll) resolviendo Perspective-n-Point.
    Retorna: (pitch, yaw, roll, rvec, tvec) en grados.
    Pitch: Inclinación arriba/abajo (- = agachado/cabeceo de microsueño).
    Yaw: Giro izquierda/derecha (+/- = distracción de ruta).
    Roll: Inclinación lateral hacia hombros.
    """
    h, w = image_shape[:2]
    image_points = []
    for lm_id in HEAD_POSE_LANDMARK_IDS:
        if lm_id not in landmarks_dict:
            return 0.0, 0.0, 0.0, np.zeros((3, 1)), np.zeros((3, 1))
        pt = landmarks_dict[lm_id]
        image_points.append([pt[0] * w, pt[1] * h])

    image_points = np.array(image_points, dtype=np.float64)

    # Matriz intrínseca aproximada de la cámara
    focal_length = float(w)
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        FACE_3D_MODEL,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0, np.zeros((3, 1)), np.zeros((3, 1))

    # Matriz de rotación a ángulos de Euler
    rot_matrix, _ = cv2.Rodrigues(rvec)
    proj_matrix = np.hstack((rot_matrix, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    pitch = float(euler_angles[0, 0])
    yaw = float(euler_angles[1, 0])
    roll = float(euler_angles[2, 0])

    return pitch, yaw, roll, rvec, tvec
