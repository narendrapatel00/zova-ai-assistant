"""
Text-to-Speech High-Level Service for ZovaAI.
Wraps the SpeechSynthesizer interface to coordinate vocalization.
"""

from pathlib import Path
from src.interfaces.speech_synthesizer import SpeechSynthesizer


class SpeechSynthesisService:
    """Service class encapsulating speech synthesis and vocalization operations."""

    def __init__(self, synthesizer: SpeechSynthesizer):
        """
        Initializes the service with a SpeechSynthesizer.

        Args:
            synthesizer: Resolved SpeechSynthesizer singleton.
        """
        self.synthesizer = synthesizer

    def speak(self, text: str) -> None:
        """
        Synthesizes the text and plays it back to the user immediately.

        Args:
            text: Sentence to speak.
        """
        self.synthesizer.speak(text)

    def synthesize_to_file(self, text: str, output_path: Path) -> Path:
        """
        Synthesizes text and saves the output to a specified WAV file.

        Args:
            text: Sentence to synthesize.
            output_path: Path to write the output WAV.

        Returns:
            Path: The generated output file path.
        """
        return self.synthesizer.synthesize(text, output_path)
