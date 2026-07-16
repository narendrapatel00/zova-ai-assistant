"""
Wake Word Listening Service Implementation for ZovaAI.
Runs an independent background thread, subscribes to AudioRecorder via the
Observer pattern to receive frame notifications, and triggers callbacks.
"""

import time
import queue
import threading
import wave
from pathlib import Path
from typing import Optional, Callable
import numpy as np

from src.core.config import Config
from src.core.logger import get_logger
from src.core.exceptions import WakeWordError, AudioError
from src.interfaces.audio_recorder import AudioRecorder, AudioListener
from src.interfaces.wake_word_detector import WakeWordDetector
from src.audio.playback import play_wav
from src.wakeword.interfaces import WakeWordService

logger = get_logger("wakeword_service")


class WakeWordListeningService(WakeWordService, AudioListener):
    """Background listener service coordinating audio stream capture and wake word detection."""

    def __init__(
        self,
        config: Config,
        audio_recorder: AudioRecorder,
        detector: WakeWordDetector
    ):
        """
        Initializes the wake word service.

        Args:
            config: Loaded application configuration manager.
            audio_recorder: Resolved AudioRecorder singleton.
            detector: Resolved WakeWordDetector singleton.
        """
        self.config = config
        self.audio_recorder = audio_recorder
        self.detector = detector
        self.cooldown = config.wakeword.cooldown_seconds
        self.enabled = config.wakeword.enabled

        self._incoming_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[], None]] = None
        self._lock = threading.Lock()

        # Ensure activation sound WAV exists locally
        self._sound_path = self._ensure_activation_sound()

    def _ensure_activation_sound(self) -> Path:
        """
        Generates a default offline 880Hz confirmation beep if assets are missing.

        Returns:
            Path: The file path pointing to the activation WAV.
        """
        assets_dir = self.config.project_root / "assets" / "audio"
        assets_dir.mkdir(parents=True, exist_ok=True)
        sound_path = assets_dir / "activation.wav"

        if not sound_path.exists():
            logger.info("Generating default offline activation sound at %s", sound_path)
            sample_rate = 16000
            duration = 0.15
            frequency = 880.0
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

            # Sine wave with gentle fade out to prevent speaker clicking
            audio_data = np.sin(2 * np.pi * frequency * t) * 0.25
            fade_len = int(sample_rate * 0.02)
            fade_out = np.linspace(1.0, 0.0, fade_len)
            audio_data[-fade_len:] *= fade_out

            pcm_data = (audio_data * 32767.0).astype(np.int16)

            try:
                # pylint: disable=no-member
                with wave.open(str(sound_path), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(pcm_data.tobytes())
                logger.info("Default confirmation WAV created.")
            except Exception as e:
                logger.error("Failed to generate offline activation sound: %s", e)

        return sound_path

    def register_callback(self, callback: Callable[[], None]) -> None:
        """
        Registers the event handler function to execute on wake word detection.

        Args:
            callback: Function to run without arguments.
        """
        with self._lock:
            self._callback = callback
            logger.debug("Registered wake-word detected callback.")

    def start(self) -> None:
        """Starts the background wake-word listening loop thread."""
        with self._lock:
            if self._running:
                logger.warning("WakeWordListeningService is already running.")
                return

            if not self.enabled:
                logger.info("Wake-word engine is disabled in configuration. Startup skipped.")
                return

            self._running = True
            # Clear queue before starting
            while not self._incoming_queue.empty():
                try:
                    self._incoming_queue.get_nowait()
                except queue.Empty:
                    break

            # Subscribe to the AudioRecorder frame stream
            self.audio_recorder.subscribe(self)

            self._thread = threading.Thread(
                target=self._listen_loop,
                name="WakeWordListenerThread",
                daemon=True
            )
            self._thread.start()
            logger.info("Wake-word listener service started.")

    def stop(self) -> None:
        """Stops the background listening thread gracefully."""
        thread_to_join = None
        with self._lock:
            if not self._running:
                return
            self._running = False

            # Unsubscribe from the AudioRecorder frame stream
            try:
                self.audio_recorder.unsubscribe(self)
            # pylint: disable=broad-exception-caught
            except Exception as e:
                logger.warning("Error unsubscribing from audio stream: %s", e)

            thread_to_join = self._thread
            self._thread = None

        if thread_to_join:
            logger.info("Stopping wake-word listener service...")
            # Thread will unblock on queue timeout and exit
            thread_to_join.join(timeout=2.0)
            logger.info("Wake-word listener service stopped.")

    def is_running(self) -> bool:
        """Checks if the service loop is active."""
        with self._lock:
            return self._running

    def on_audio_chunk(self, chunk: np.ndarray) -> None:
        """
        Observer callback called by AudioRecorder when a new chunk is captured.

        Args:
            chunk: 1D raw audio array chunk.
        """
        if not self.is_running():
            return

        # Queue chunk for background thread processing
        if self._incoming_queue.full():
            try:
                self._incoming_queue.get_nowait()
            except queue.Empty:
                pass
        self._incoming_queue.put(chunk)

    def _listen_loop(self) -> None:
        """Continuous background listening loop feeding chunks to openWakeWord."""
        last_trigger_time = 0.0
        logger.info("Listening for '%s'...", self.detector.get_wake_word_name())

        while True:
            with self._lock:
                if not self._running:
                    break

            try:
                # 1. Fetch latest audio block from the internal queue (blocks up to 1.0s)
                chunk = self._incoming_queue.get(timeout=1.0)

                # 2. Skip detection if Zova is currently recording a command
                if self.audio_recorder.is_recording():
                    continue

                # 3. Perform wake-word inference
                detected = self.detector.detect(chunk)
                if detected:
                    current_time = time.time()
                    if current_time - last_trigger_time >= self.cooldown:
                        logger.info("Wake-word detected! (confidence above threshold)")
                        last_trigger_time = current_time

                        # Play confirmation chirp
                        try:
                            play_wav(self._sound_path)
                        except AudioError as ae:
                            logger.error("Failed to play confirmation sound: %s", ae.message)

                        # Fire registered event handler
                        with self._lock:
                            cb = self._callback
                        if cb:
                            cb()
                    else:
                        logger.info("Wake word matched but cooldown is active.")

            except queue.Empty:
                # Queue empty is expected when there is no microphone input activity
                continue
            except WakeWordError as wwe:
                logger.error("Wake-word inference failure in listener: %s", wwe.message)
                time.sleep(0.5)
            except Exception as e:
                logger.error("Unhandled exception in wake-word listen loop: %s", e)
                time.sleep(0.5)
