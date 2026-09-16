import cv2
import threading
import time
from typing import Optional, Tuple, Union
import numpy as np


class CameraStream:
    """
    Capturador de video multihilo no bloqueante con OpenCV.
    Garantiza que el bucle de detección siempre reciba el frame más reciente sin retardo acumulativo de buffer.
    """

    def __init__(self, src: Union[int, str] = 0, width: int = 640, height: int = 480):
        self.src = src
        self.width = width
        self.height = height

        self.cap = cv2.VideoCapture(self.src)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.cap.read() if self.cap.isOpened() else (False, None)
        self.stopped = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> 'CameraStream':
        if not self.cap.isOpened():
            return self

        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                break
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed = grabbed
                if grabbed:
                    self.frame = frame
            time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()
