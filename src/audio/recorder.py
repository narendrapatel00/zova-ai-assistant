"""
Sounddevice Audio Recorder Implementation for ZovaAI.
Implements the AudioRecorder interface contract. Runs a background input stream,
supports RMS-based VAD auto-stop, and exports recorded voice as 16-bit WAV PCM.
Supports registering subscribers (AudioListener) using the Observer pattern.
"""

import queue
import threading
import uuid
import wave
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]

from src.core.config import Config
from src.core.logger import get_logger
from src.core.exceptions import AudioError
from src.interfaces.audio_recorder import AudioRecorder, AudioListener

logger = get_logger("audio_recorder")


# pylint: disable=too-many-instance-attributes
class SounddeviceAudioRecorder(AudioRecorder):
    """Concrete implementation of AudioRecorder interface using sounddevice."""

    def __init__(self, config: Config):
        """
        Initializes the audio recorder.
        
        Args:
            config: Loaded application configuration manager.
        """
        self.config = config
        self.sample_rate = config.audio.sample_rate
        self.channels = config.audio.channels
        self.chunk_size = config.audio.chunk_size
        self.device_index = config.audio.device_index
        self.silence_threshold = config.audio.silence_threshold
        self.silence_seconds = config.audio.silence_seconds

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._recording_buffer: List[np.ndarray] = []
        self._listeners: List[AudioListener] = []
        self._is_recording = False
        self._has_speech_started = False
        self._silence_duration = 0.0
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

        # Initialize the background stream
        self._init_stream()

    @staticmethod
    def list_devices() -> List[Dict[str, Any]]:
        """
        Lists all available audio input devices (microphones).
        
        Returns:
            List[Dict[str, Any]]: List of device dictionaries containing index and info.
            
        Raises:
            AudioError: If device enumeration fails.
        """
        try:
            devices = sd.query_devices()
            input_devices = []
            for i, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0:
                    input_devices.append({
                        "index": i,
                        "name": dev["name"],
                        "max_input_channels": dev["max_input_channels"],
                        "default_samplerate": dev["default_samplerate"]
                    })
            return input_devices
        except Exception as e:
            raise AudioError(f"Failed to query audio devices: {str(e)}") from e

    def _init_stream(self) -> None:
        """Initializes and starts the background sounddevice InputStream."""
        try:
            devices = sd.query_devices()
            
            # Select target device index
            target_device = self.device_index
            if target_device is None:
                default_device = sd.default.device[0]
                if default_device < 0:
                    raise AudioError("No default input audio device found on the system.")
                target_device = default_device

            # Validate index
            if target_device < 0 or target_device >= len(devices):
                raise AudioError(f"Invalid audio device index: {target_device}")

            device_info = devices[target_device]
            if device_info.get("max_input_channels", 0) <= 0:
                raise AudioError(
                    f"Selected device {target_device} ({device_info['name']}) "
                    f"has no input channels."
                )

            logger.info(
                "Starting InputStream on device index %s (%s) at %sHz",
                target_device, device_info["name"], self.sample_rate
            )

            # Start InputStream
            self._stream = sd.InputStream(
                device=target_device,
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self._audio_callback,
                blocksize=self.chunk_size,
                dtype="float32"
            )
            self._stream.start()
            logger.info("Audio InputStream active.")

        except Exception as e:
            if not isinstance(e, AudioError):
                raise AudioError(f"Failed to initialize audio stream: {str(e)}") from e
            raise e

    def _audio_callback(
        self, indata: np.ndarray, frames: int, _time_info: Any, status: sd.CallbackFlags
    ) -> None:
        """Background callback function that receives data chunks from sounddevice."""
        if status:
            logger.warning("Sounddevice status flag: %s", status)

        # Make copy of array to prevent buffer reuse overrides
        chunk = indata.copy()

        # If queue is full, discard oldest chunk to prevent memory bloat
        if self._audio_queue.full():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                pass
        self._audio_queue.put(chunk)

        # Process VAD, buffer recording, and list copy
        with self._lock:
            if self._is_recording:
                flat_chunk = chunk.flatten()
                self._recording_buffer.append(flat_chunk)

                # Compute RMS energy level
                rms = np.sqrt(np.mean(flat_chunk**2))

                if rms >= self.silence_threshold:
                    self._has_speech_started = True
                    self._silence_duration = 0.0
                else:
                    if self._has_speech_started:
                        self._silence_duration += frames / self.sample_rate
                        if self._silence_duration >= self.silence_seconds:
                            logger.info(
                                "Silence detected for %.2fs. Auto-stopping.",
                                self._silence_duration
                            )
                            self._is_recording = False
            
            # Copy active listener references
            listeners_copy = self._listeners.copy()

        # Notify subscribers
        flat_chunk_for_listeners = chunk.flatten()
        for listener in listeners_copy:
            try:
                listener.on_audio_chunk(flat_chunk_for_listeners)
            except Exception as e:
                logger.error("Error in audio listener callback: %s", e)

    def start_recording(self) -> None:
        """Starts recording audio from background stream."""
        with self._lock:
            if self._is_recording:
                raise AudioError("AudioRecorder is already recording.")
            
            # Verify stream is healthy
            if not self._stream or not self._stream.active:
                raise AudioError("Cannot record: Audio InputStream is inactive.")

            self._recording_buffer.clear()
            self._silence_duration = 0.0
            self._has_speech_started = False
            self._is_recording = True
            logger.info("Microphone recording started.")

    def stop_recording(self) -> Path:
        """Stops recording session and returns file path to WAV format."""
        with self._lock:
            if not self._recording_buffer:
                raise AudioError("No audio data was recorded. Call start_recording first.")

            self._is_recording = False
            # Flatten lists to 1D numpy array
            audio_data = np.concatenate(self._recording_buffer)
            self._recording_buffer.clear()

        return self._save_wav(audio_data)

    def get_audio_chunk(self) -> np.ndarray:
        """Retrieves latest audio block chunk."""
        if not self._stream or not self._stream.active:
            raise AudioError("Audio stream is not active.")

        try:
            chunk = self._audio_queue.get(timeout=1.0)
            return chunk.flatten()
        except queue.Empty as e:
            raise AudioError("Timed out waiting for audio chunk from microphone.") from e

    def is_recording(self) -> bool:
        """Checks if recording is currently active."""
        with self._lock:
            return self._is_recording

    def subscribe(self, listener: AudioListener) -> None:
        """Registers an AudioListener to receive real-time audio frames."""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)
                logger.debug("Subscribed new audio listener: %s", type(listener).__name__)

    def unsubscribe(self, listener: AudioListener) -> None:
        """Unsubscribes a previously registered AudioListener."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)
                logger.debug("Unsubscribed audio listener: %s", type(listener).__name__)

    def _save_wav(self, audio_data: np.ndarray) -> Path:
        """Saves float32 audio data array to 16-bit PCM mono WAV."""
        temp_dir = self.config.project_root / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        filename = f"recording_{uuid.uuid4().hex}.wav"
        file_path = temp_dir / filename

        # Clip float range to prevent overflow noise, scale to 16-bit
        clipped = np.clip(audio_data, -1.0, 1.0)
        pcm_data = (clipped * 32767.0).astype(np.int16)

        try:
            # Write frames using wave module
            # pylint: disable=no-member
            with wave.open(str(file_path), "wb") as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 2 bytes = 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(pcm_data.tobytes())

            logger.info("Recording saved to %s", file_path)
            return file_path
        except Exception as e:
            raise AudioError(f"Failed to write audio WAV format file: {str(e)}") from e

    def close(self) -> None:
        """Closes background streams and releases audio cards."""
        with self._lock:
            self._is_recording = False
            self._listeners.clear()
        
        logger.info("Releasing audio streams...")
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            # pylint: disable=broad-exception-caught
            except Exception as e:
                logger.warning("Error closing sounddevice stream: %s", e)
        logger.info("Audio stream closed.")
