"""
Event Subsystem Package for ZovaAI.
Contains FIFO asynchronous event buses and strongly typed pipeline events.
"""

from src.events.event_bus import EventBus
from src.events.events import (
    Event,
    WakeWordDetected,
    RecordingStarted,
    RecordingFinished,
    STTCompleted,
    LLMStarted,
    LLMChunkReceived,
    LLMCompleted,
    TTSStarted,
    TTSCompleted,
    ErrorOccurred
)

__all__ = [
    "EventBus",
    "Event",
    "WakeWordDetected",
    "RecordingStarted",
    "RecordingFinished",
    "STTCompleted",
    "LLMStarted",
    "LLMChunkReceived",
    "LLMCompleted",
    "TTSStarted",
    "TTSCompleted",
    "ErrorOccurred"
]
