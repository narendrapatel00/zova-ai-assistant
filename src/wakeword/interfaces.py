"""
Abstract Interfaces for Wake Word Subsystem.
Defines contracts for the background wake-word listener service.
"""

from abc import ABC, abstractmethod


class WakeWordService(ABC):
    """Abstract interface defining the background wake-word listening lifecycle service."""

    @abstractmethod
    def start(self) -> None:
        """
        Starts the background wake-word listening thread.
        Loads the underlying models and hooks into the audio capture stream.
        """

    @abstractmethod
    def stop(self) -> None:
        """
        Stops the background listening thread and releases locks.
        """

    @abstractmethod
    def is_running(self) -> bool:
        """
        Checks if the background listening thread is active.
        
        Returns:
            bool: True if running, False otherwise.
        """
