"""
Piper Text-to-Speech (TTS) Synthesizer Implementation for ZovaAI.
Implements the SpeechSynthesizer interface contract. Runs the local pre-compiled
Piper executable to generate standard voice WAV files and play them back.
"""

import os
import subprocess
import time
import uuid
from pathlib import Path

from src.core.config import Config
from src.core.logger import get_logger
from src.core.exceptions import TTSError, AudioError
from src.interfaces.speech_synthesizer import SpeechSynthesizer
from src.audio.playback import play_wav

logger = get_logger("speech_synthesizer")


class PiperSpeechSynthesizer(SpeechSynthesizer):
    """Text-to-Speech synthesizer implementation using the precompiled Piper engine."""

    def __init__(self, config: Config):
        """
        Initializes the Piper speech synthesizer.

        Args:
            config: Loaded application configuration manager.

        Raises:
            TTSError: If binaries or model ONNX files are missing.
        """
        self.config = config
        self.enabled = config.tts.enabled
        self.executable_path = config.tts.executable_path
        self.model_path = config.tts.model_path
        self.config_path = config.tts.config_path
        self.output_dir = config.tts.output_dir

        if self.enabled:
            self._verify_resources()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Piper speech synthesizer initialized successfully.")
        else:
            logger.info("Piper speech synthesizer is disabled in configuration.")

    def _verify_resources(self) -> None:
        """Verifies that all Piper executables, ONNX models, and configs exist on disk."""
        if not self.executable_path.exists():
            raise TTSError(f"Piper executable not found at: {self.executable_path}")
        if not self.model_path.exists():
            raise TTSError(f"Piper ONNX model not found at: {self.model_path}")
        if not self.config_path.exists():
            raise TTSError(f"Piper ONNX voice configuration not found at: {self.config_path}")

    def synthesize(self, text: str, output_path: Path) -> Path:
        """
        Synthesizes plain text into a WAV file using the local Piper executable.

        Args:
            text: Sentence string to read.
            output_path: Target Path where WAV format output is written.

        Returns:
            Path: The generated output file path.

        Raises:
            TTSError: If subprocess execution fails or file is not written.
        """
        if not self.enabled:
            raise TTSError("Piper SpeechSynthesizer is disabled.")

        if not text.strip():
            raise TTSError("Speech synthesis text input cannot be empty.")

        try:
            logger.info("Speech synthesis started for text: \"%s\"", text)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Setup subprocess command
            cmd = [
                str(self.executable_path),
                "--model", str(self.model_path),
                "--config", str(self.config_path),
                "--output_file", str(output_path)
            ]

            start_time = time.time()

            # Run Piper subprocess, feeding text to standard input
            # We enforce quiet mode (-q) to prevent Piper printing debug text to stderr
            cmd.append("-q")

            result = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                check=False
            )

            if result.returncode != 0:
                error_msg = (
                    result.stderr.decode("utf-8", errors="ignore").strip()
                )
                raise TTSError(
                    f"Piper synthesis process exited with code "
                    f"{result.returncode}: {error_msg}"
                )

            latency = time.time() - start_time
            logger.info("Piper synthesis complete (%.2fs)", latency)

            # Check output file
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise TTSError("Piper process succeeded but output WAV file is missing or empty.")

            return output_path

        except Exception as e:
            if not isinstance(e, TTSError):
                raise TTSError(f"Piper subprocess execution failed: {str(e)}") from e
            raise e

    def speak(self, text: str) -> None:
        """
        Synthesizes text and plays it back to the user immediately.
        Cleans up temporary WAV files generated during playback.

        Args:
            text: Sentence string to speak.

        Raises:
            TTSError: If synthesis or playbacks fail.
        """
        temp_wav = self.output_dir / f"speech_{uuid.uuid4().hex}.wav"

        try:
            # 1. Synthesize text
            self.synthesize(text, temp_wav)

            # 2. Play WAV
            play_wav(temp_wav, device_index=self.config.audio.device_index)

        except AudioError as ae:
            raise TTSError(f"Failed to play back synthesized speech: {ae.message}") from ae
        finally:
            # 3. Clean up the temporary file
            try:
                if temp_wav.exists():
                    os.unlink(temp_wav)
            # pylint: disable=broad-exception-caught
            except Exception as e:
                logger.warning(
                    "Failed to delete temporary speech WAV file '%s': %s",
                    temp_wav.name, e
                )

    def close(self) -> None:
        """Releases resource hooks and log release."""
        logger.info("Releasing Piper speech synthesizer context.")
