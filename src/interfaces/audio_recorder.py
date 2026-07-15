"""
Abstract Interface for Audio Recorder and Listeners.
Defines contracts for real-time audio chunk capture, audio recording storage,
and the Observer (Pub/Sub) subscription pattern.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class AudioListener(ABC):
    """Abstract interface defining the contract for audio stream consumers."""

    @abstractmethod
    def on_audio_chunk(self, chunk: np.ndarray) -> None:
        """
        Callback triggered when a new raw audio chunk is captured.
        
        Args:
            chunk: 1D numpy array representing 16kHz mono audio samples.
        """


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
    def subscribe(self, listener: AudioListener) -> None:
        """
        Subscribes an AudioListener to receive real-time audio frames.
        
        Args:
            listener: Concrete listener subclass instance.
        """

    @abstractmethod
    def unsubscribe(self, listener: AudioListener) -> None:
        """
        Unsubscribes a previously registered AudioListener.
        
        Args:
            listener: Concrete listener subclass instance.
        """

    @abstractmethod
    def close(self) -> None:
        """
        Closes the audio recording streams and releases hardware resource hooks.
        """
