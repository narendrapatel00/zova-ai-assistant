"""
Abstract Interface for Audio Recorder.
Defines contracts for real-time audio chunk capture and audio recording storage.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class AudioRecorder(ABC):
    """Abstract base class defining the contract for recording voice commands."""

    @abstractmethod
    def start_recording(self) -> None:
        """
        Starts recording audio. Must initialize hardware buffers if not already open.
        
        Raises:
            AudioError: If recorder is already recording or hardware fails.
        """

    @abstractmethod
    def stop_recording(self) -> Path:
        """
        Stops the recording session and flushes buffer contents to a WAV file.
        
        Returns:
            Path: The file path pointing to the saved WAV recording.
            
        Raises:
            AudioError: If not currently recording or file creation fails.
        """

    @abstractmethod
    def get_audio_chunk(self) -> np.ndarray:
        """
        Fetches the latest block of audio data from the microphone.
        Used for real-time wake word detection stream feed.
        
        Returns:
            np.ndarray: Audio data as 1D array of 16-bit 16kHz PCM samples.
            
        Raises:
            AudioError: If microphone stream fails or device is disconnected.
        """

    @abstractmethod
    def is_recording(self) -> bool:
        """
        Checks if the recorder is currently capturing a voice command.
        
        Returns:
            bool: True if recording, False otherwise.
        """

    @abstractmethod
    def close(self) -> None:
        """
        Closes the audio recording streams and releases hardware resource hooks.
        """
