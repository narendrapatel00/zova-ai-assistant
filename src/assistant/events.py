"""
Event Observer Contracts for ZovaAI Assistant Orchestrator.
Defines listener lifecycle events for assistant states.
"""

from abc import ABC
from pathlib import Path


class AssistantEventObserver(ABC):
    """Abstract interface defining the hook callbacks for assistant events."""

    def on_waiting_for_wake_word(self) -> None:
        """Fired when assistant enters idle state waiting for Jarvis wake trigger."""

    def on_wake_word_detected(self) -> None:
        """Fired when Jarvis wake word is matched."""

    def on_recording_started(self) -> None:
        """Fired when user command voice recording session begins."""

    def on_recording_finished(self, wav_path: Path) -> None:
        """Fired when VAD silence stops recording and outputs WAV format."""

    def on_transcription_finished(self, text: str) -> None:
        """Fired when speech-to-text transcription completes."""

    def on_llm_start(self, prompt: str) -> None:
        """Fired when prompt payload is dispatched to local Ollama server."""

    def on_llm_response(self, response: str) -> None:
        """Fired when Ollama returns assistant text reply."""

    def on_tts_start(self, text: str) -> None:
        """Fired when TTS starts vocalizing response text."""

    def on_tts_finished(self) -> None:
        """Fired when response voice playback terminates."""

    def on_error(self, error_message: str) -> None:
        """Fired when pipeline crash or device disconnect failures are caught."""
