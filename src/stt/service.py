"""
Speech-to-Text High-Level Service for ZovaAI.
Wraps the SpeechRecognizer interface to coordinate transcriptions.
"""

from pathlib import Path
from src.interfaces.speech_recognizer import SpeechRecognizer


class SpeechRecognitionService:
    """Service class encapsulating speech-to-text operations."""

    def __init__(self, recognizer: SpeechRecognizer):
        """
        Initializes the service with a SpeechRecognizer.

        Args:
            recognizer: Resolved SpeechRecognizer singleton.
        """
        self.recognizer = recognizer

    def transcribe_audio(self, audio_path: Path) -> str:
        """
        Transcribes the speech recorded in the specified audio WAV file to text.

        Args:
            audio_path: Absolute Path to the WAV file.

        Returns:
            str: Transcribed text output.
        """
        return self.recognizer.transcribe(audio_path)
