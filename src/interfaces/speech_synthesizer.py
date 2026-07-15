"""
Abstract Interface for Speech Synthesizer.
Defines contracts for offline text-to-speech synthesis and voice playback.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class SpeechSynthesizer(ABC):
    """Abstract base class defining the contract for local speech synthesis engines."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """
        Synthesizes the text into speech and plays it back to the user immediately.
        
        Args:
            text: The text sentence to read aloud.
            
        Raises:
            TTSError: If synthesis fails or audio hardware cannot play back the file.
        """

    @abstractmethod
    def synthesize(self, text: str, output_path: Path) -> Path:
        """
        Synthesizes the text and saves the output to a specified WAV file path.
        
        Args:
            text: The text sentence to synthesize.
            output_path: Target Path to save the output WAV file.
            
        Returns:
            Path: The path to the created WAV file.
            
        Raises:
            TTSError: If synthesis fails.
        """

    @abstractmethod
    def close(self) -> None:
        """
        Releases synthesis engine hooks and temporary file references.
        """
