"""
Custom Exception Classes for ZovaAI.
Defines a robust hierarchy of exceptions following OOP and SOLID practices.
"""

class ZovaException(Exception):
    """Base exception class for all errors in ZovaAI."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ConfigurationError(ZovaException):
    """Raised when application configuration loading or validation fails."""


class AudioError(ZovaException):
    """Raised when audio recording, playing, or hardware initialization fails."""


class WakeWordError(ZovaException):
    """Raised when the wake word detection engine fails to initialize or process."""


class STTError(ZovaException):
    """Raised when speech-to-text initialization or transcription fails."""


class TTSError(ZovaException):
    """Raised when text-to-speech voice synthesis or audio playback fails."""


class DependencyInjectionError(ZovaException):
    """Raised when there is an issue registering or resolving dependencies."""


class SetupError(ZovaException):
    """Raised when download or setup of binary tools and models fails."""
