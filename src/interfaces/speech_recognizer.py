"""
Abstract Interface for Speech Recognizer.
Defines contracts for offline speech-to-text transcription.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class SpeechRecognizer(ABC):
    """Abstract base class defining the contract for local speech transcription engines."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> str:
        """
        Transcribes the speech recorded in the specified audio WAV file to plain text.
        
        Args:
            audio_path: Absolute Path to the WAV audio file.
            
        Returns:
            str: Transcribed text output.
            
        Raises:
            STTError: If transcription execution or model processing fails.
        """

    @abstractmethod
    def close(self) -> None:
        """
        Releases whisper models and engine hooks to free memory resources.
        """
