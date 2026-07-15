"""
Abstract Interface for Wake Word Detector.
Defines contracts for listening to "Hey Jarvis" triggers in real-time.
"""

from abc import ABC, abstractmethod
import numpy as np


class WakeWordDetector(ABC):
    """Abstract base class defining the contract for wake word engines."""

    @abstractmethod
    def detect(self, chunk: np.ndarray) -> bool:
        """
        Processes a single buffer chunk of audio to detect the configured wake word.
        
        Args:
            chunk: A numpy array representing 16-bit 16kHz PCM audio samples.
            
        Returns:
            bool: True if the wake word is detected with confidence, False otherwise.
            
        Raises:
            WakeWordError: If inference calculation fails.
        """

    @abstractmethod
    def get_wake_word_name(self) -> str:
        """
        Gets the target wake word string (e.g., 'hey jarvis').
        
        Returns:
            str: Target wake word name.
        """

    @abstractmethod
    def close(self) -> None:
        """
        Releases models and engine hooks to free memory resources.
        """
