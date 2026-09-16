import threading
import time
import numpy as np
from typing import Dict, Any, Optional
from src.core.early_warning import DriverAlertState

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class AlarmManager:
    """
    Gestor de alarmas sonoras asíncronas no bloqueantes.
    Sintetiza tonos PCM en memoria (no requiere archivos WAV externos obligatorios).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("alerts", {}).get("enable_audio", True)
        self.audio_initialized = False
        self.last_played_state: Optional[DriverAlertState] = None
        self.last_play_time: float = 0.0
        self.is_playing_critical = False
        self.stop_event = threading.Event()

        if self.enabled and PYGAME_AVAILABLE:
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
                self.audio_initialized = True
                self._generate_sound_buffers()
            except Exception:
                self.audio_initialized = False

    def _generate_sound_buffers(self):
        """Genera ondas sinusoidales puras en memoria para las diferentes alertas."""
        sample_rate = 44100

        # Tono suave de atención (880 Hz, 150 ms)
        t_att = np.linspace(0, 0.15, int(sample_rate * 0.15), False)
        tone_att = (np.sin(2 * np.pi * 880 * t_att) * 15000).astype(np.int16)
        self.sound_attention = pygame.sndarray.make_sound(tone_att)

        # Tono de advertencia doble (600 Hz y 750 Hz, 300 ms)
        t_warn = np.linspace(0, 0.30, int(sample_rate * 0.30), False)
        tone_warn = (np.sin(2 * np.pi * 650 * t_warn) * 20000).astype(np.int16)
        self.sound_warning = pygame.sndarray.make_sound(tone_warn)

        # Tono crítico penetrante (1200 Hz con modulación pulsante)
        t_crit = np.linspace(0, 0.40, int(sample_rate * 0.40), False)
        tone_crit = (np.sin(2 * np.pi * 1200 * t_crit) * 30000).astype(np.int16)
        self.sound_critical = pygame.sndarray.make_sound(tone_crit)

    def trigger_alert(self, state: DriverAlertState, score: float):
        """Dispara la alarma correspondiente en un hilo independiente según la severidad."""
        if not self.enabled or not self.audio_initialized:
            return

        now = time.time()

        if state == DriverAlertState.CRITICAL:
            # En estado crítico, insistir con sonido cada 0.5 segundos
            if now - self.last_play_time > 0.45:
                threading.Thread(target=self._play_sound, args=(self.sound_critical,), daemon=True).start()
                self.last_play_time = now
                self.last_played_state = state

        elif state == DriverAlertState.WARNING:
            # Advertencia: espaciado de al menos 3.0 segundos
            if now - self.last_play_time > 3.0:
                threading.Thread(target=self._play_sound, args=(self.sound_warning,), daemon=True).start()
                self.last_play_time = now
                self.last_played_state = state

        elif state == DriverAlertState.ATTENTION or state == DriverAlertState.DISTRACTED:
            # Atención temprana: espaciado de al menos 4.0 segundos
            if now - self.last_play_time > 4.0:
                threading.Thread(target=self._play_sound, args=(self.sound_attention,), daemon=True).start()
                self.last_play_time = now
                self.last_played_state = state

    def _play_sound(self, sound_obj):
        """Reproducción en hilo aislado para cero impacto en los FPS de video."""
        try:
            sound_obj.play()
        except Exception:
            pass

    def cleanup(self):
        """Libera recursos del mezclador de audio."""
        if self.audio_initialized and PYGAME_AVAILABLE:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
