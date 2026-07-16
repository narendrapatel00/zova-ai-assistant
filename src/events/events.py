"""
Event definitions for ZovaAI.
Defines strongly typed event dataclasses for pipeline events.
"""

from dataclasses import dataclass
from pathlib import Path


class Event:
    """Base class for all events in ZovaAI."""


@dataclass
class WakeWordDetected(Event):
    """Fired when the wake word is matched."""


@dataclass
class RecordingStarted(Event):
    """Fired when voice command recording begins."""


@dataclass
class RecordingFinished(Event):
    """Fired when command recording ends and audio is saved to disk."""
    wav_path: Path


@dataclass
class STTCompleted(Event):
    """Fired when Speech-to-Text transcription completes."""
    text: str


@dataclass
class LLMStarted(Event):
    """Fired when user query is sent to local Ollama server."""
    prompt: str


@dataclass
class LLMChunkReceived(Event):
    """Fired when an LLM streaming token chunk is received."""
    chunk: str


@dataclass
class LLMCompleted(Event):
    """Fired when local Ollama returns complete response text."""
    response: str


@dataclass
class TTSStarted(Event):
    """Fired when Text-to-Speech synthesis begins."""
    text: str


@dataclass
class TTSCompleted(Event):
    """Fired when the synthesized voice playback completes."""


@dataclass
class ErrorOccurred(Event):
    """Fired when any pipeline execution crash or failure is encountered."""
    error_message: str
