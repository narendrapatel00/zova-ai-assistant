"""
Core Utilities Package for ZovaAI.
Contains configuration, logging, custom exceptions, and dependency injection container.
"""

from src.core.config import Config
from src.core.di import DIContainer
from src.core.exceptions import (
    ZovaException,
    ConfigurationError,
    AudioError,
    WakeWordError,
    STTError,
    TTSError,
    DependencyInjectionError,
    SetupError
)
from src.core.logger import LoggerSetup, get_logger

__all__ = [
    "Config",
    "DIContainer",
    "ZovaException",
    "ConfigurationError",
    "AudioError",
    "WakeWordError",
    "STTError",
    "TTSError",
    "DependencyInjectionError",
    "SetupError",
    "LoggerSetup",
    "get_logger"
]
