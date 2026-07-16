"""
Whisper.cpp Speech-to-Text Recognizer Implementation for ZovaAI.
Implements the SpeechRecognizer interface. Performs local, offline transcription
using the pywhispercpp GGML-based engine.
"""

import time
import wave
from pathlib import Path
from typing import Optional
from pywhispercpp.model import Model  # type: ignore[import-untyped]

from src.core.config import Config
from src.core.logger import get_logger
from src.core.exceptions import STTError
from src.interfaces.speech_recognizer import SpeechRecognizer

logger = get_logger("speech_recognizer")


# pylint: disable=too-many-instance-attributes
class WhisperSpeechRecognizer(SpeechRecognizer):
    """SpeechRecognizer implementation using Whisper.cpp (pywhispercpp)."""

    def __init__(self, config: Config):
        """
        Initializes the Whisper model context.

        Args:
            config: Loaded application configuration manager.

        Raises:
            STTError: If model cannot be loaded or initialized.
        """
        self.config = config
        self.model_path = config.stt.model_path
        self.enabled = config.stt.enabled
        self.threads = config.stt.threads
        self.language = config.stt.language
        self.beam_size = config.stt.beam_size
        self.temperature = config.stt.temperature

        self._model: Optional[Model] = None

        if self.enabled:
            self._load_model()
        else:
            logger.info("STT Whisper recognizer is disabled in configuration.")

    def _load_model(self) -> None:
        """Loads the Whisper model from model_path into memory."""
        try:
            resolved_path = self.config.resolve_path(str(self.model_path))
            if not resolved_path.exists():
                raise STTError(f"Whisper model file not found: {resolved_path}")

            logger.info("Loading Whisper.cpp model from: %s", resolved_path)
            start_time = time.time()

            # Initialize model context (reused across requests)
            self._model = Model(
                str(resolved_path),
                n_threads=self.threads,
                redirect_whispercpp_logs_to=False
            )

            latency = time.time() - start_time
            logger.info("Whisper model loaded successfully (%.2fs)", latency)

        except Exception as e:
            if not isinstance(e, STTError):
                raise STTError(f"Whisper initialization failure: {str(e)}") from e
            raise e

    def transcribe(self, audio_path: Path) -> str:
        """
        Transcribes the speech recorded in the specified WAV file.

        Args:
            audio_path: Absolute Path to 16kHz mono WAV file.

        Returns:
            str: Transcribed text output.

        Raises:
            STTError: If transcription execution or model processing fails.
        """
        if not self.enabled:
            raise STTError("SpeechRecognizer is disabled.")

        if not self._model:
            raise STTError("Whisper speech recognizer is not initialized.")

        # 1. Verify file exists
        if not audio_path.exists():
            raise STTError(f"Audio file to transcribe not found: {audio_path}")

        # 2. Verify WAV properties and channel counts
        try:
            with wave.open(str(audio_path), "rb") as w:
                n_channels = w.getnchannels()
                sample_rate = w.getframerate()
                n_frames = w.getnframes()

                if sample_rate != 16000:
                    raise STTError(
                        f"Unsupported sample rate: {sample_rate}Hz. "
                        f"Whisper requires 16000Hz mono PCM."
                    )
                if n_channels != 1:
                    raise STTError(
                        f"Unsupported channels count: {n_channels}. "
                        f"Whisper requires mono PCM."
                    )
                if n_frames == 0:
                    raise STTError("Empty recording. Audio WAV file has 0 frames.")
        except Exception as e:
            if not isinstance(e, STTError):
                raise STTError(f"Invalid WAV file format or header: {str(e)}") from e
            raise e

        # 3. Execute transcription
        try:
            logger.info("Transcription started on file: %s", audio_path.name)
            start_time = time.time()

            # Execute transcription parameters
            segments = self._model.transcribe(
                str(audio_path),
                language=self.language,
                beam_size=self.beam_size,
                temperature=self.temperature
            )

            latency = time.time() - start_time
            transcribed_text = "".join([s.text for s in segments]).strip()

            logger.info("Transcription completed (%.2fs)", latency)
            logger.info("Result: \"%s\"", transcribed_text)

            return transcribed_text

        except Exception as e:
            raise STTError(f"Whisper transcription failed: {str(e)}") from e

    def close(self) -> None:
        """Releases whisper models and engine hooks to free memory."""
        logger.info("Releasing Whisper speech recognizer model context...")
        self._model = None
