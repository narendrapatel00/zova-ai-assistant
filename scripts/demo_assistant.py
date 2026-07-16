"""
Assistant Orchestrator Subsystem CLI Demo for ZovaAI.
Launches the continuous wake-word listening background thread, handles VAD recording
silence timeouts, transcribes speech locally using Whisper.cpp, queries a local Ollama
LLM model, and vocalizes the responses using the offline Piper engine.
"""

# pylint: disable=broad-exception-caught

import sys
import time

from src.core.logger import LoggerSetup, get_logger
from src.main import ZovaApp
from src.assistant.orchestrator import AssistantOrchestrator
from src.events.event_bus import EventBus
from src.events.events import (
    WakeWordDetected,
    RecordingStarted,
    RecordingFinished,
    STTCompleted,
    LLMStarted,
    LLMCompleted,
    TTSStarted,
    TTSCompleted,
    ErrorOccurred
)

logger = get_logger("demo_assistant")


class ConsoleAssistantObserver:
    """Event observer that logs state transitions directly to the console in response to events."""

    def __init__(self, event_bus: EventBus):
        """
        Initializes the observer.
        
        Args:
            event_bus: Decoupled event bus.
        """
        self.event_bus = event_bus

    def register(self) -> None:
        """Subscribes all console print callbacks to the event bus."""
        self.event_bus.subscribe(WakeWordDetected, self.on_wake_word_detected)
        self.event_bus.subscribe(RecordingStarted, self.on_recording_started)
        self.event_bus.subscribe(RecordingFinished, self.on_recording_finished)
        self.event_bus.subscribe(STTCompleted, self.on_stt_completed)
        self.event_bus.subscribe(LLMStarted, self.on_llm_start)
        self.event_bus.subscribe(LLMCompleted, self.on_llm_completed)
        self.event_bus.subscribe(TTSStarted, self.on_tts_start)
        self.event_bus.subscribe(TTSCompleted, self.on_tts_completed)
        self.event_bus.subscribe(ErrorOccurred, self.on_error)

    def on_wake_word_detected(self, event: WakeWordDetected) -> None:
        print("\n>>> [Wake Word Detected] Jarvis matched! <<<")

    def on_recording_started(self, event: RecordingStarted) -> None:
        print("[Record] Microphone active. Speak your voice command now...")

    def on_recording_finished(self, event: RecordingFinished) -> None:
        print(f"[Record] Sound captured. Exported WAV file: {event.wav_path.name}")

    def on_stt_completed(self, event: STTCompleted) -> None:
        print(f"[STT] Transcribed text: \"{event.text}\"")

    def on_llm_start(self, event: LLMStarted) -> None:
        print("[LLM] Dispatched query to local Ollama server...")

    def on_llm_completed(self, event: LLMCompleted) -> None:
        print(f"[LLM] Brain response: \"{event.response}\"")

    def on_tts_start(self, event: TTSStarted) -> None:
        print("[TTS] Running offline speech synthesis...")

    def on_tts_completed(self, event: TTSCompleted) -> None:
        print("[TTS] Vocalization playback complete.")
        print("\n-------------------------------------------------------")
        print(" [Idle] Ready. Listening for wake word: 'Jarvis'...")
        print("-------------------------------------------------------")

    def on_error(self, event: ErrorOccurred) -> None:
        print(f"\n[ERROR] Pipeline exception encountered: {event.error_message}")


def main() -> None:
    """Entry point for the offline assistant coordination demo."""
    print("\n=======================================================")
    print("        ZOVA AI ASSISTANT COORDINATION LOOP")
    print("=======================================================")

    # 1. Initialize Zova App and DI mappings
    app = ZovaApp()
    try:
        app.initialize()
        # Set console logs to INFO for clean interface formatting
        LoggerSetup.initialize(log_level="INFO")
    except Exception as e:
        print(f"FATAL: Application bootstrap failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Resolve orchestrator and event bus from DI Container
    try:
        orchestrator = app.container.resolve(AssistantOrchestrator)
        event_bus = app.container.resolve(EventBus)
    except Exception as e:
        print(f"FATAL: DI dependency resolution failed: {e}", file=sys.stderr)
        app.close()
        sys.exit(1)

    # 3. Instantiate and register the console events observer
    observer = ConsoleAssistantObserver(event_bus)
    observer.register()

    print("\nStarting orchestrator services...")
    print("Press Ctrl+C to stop the assistant and shutdown.")
    print("\n-------------------------------------------------------")
    print(" [Idle] Ready. Listening for wake word: 'Jarvis'...")
    print("-------------------------------------------------------")

    try:
        # Start the background pipeline
        orchestrator.start()
        
        # Keep the main thread alive while background workers capture audio
        while orchestrator.is_running():
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nShutting down assistant...")
    finally:
        orchestrator.stop()
        app.close()
        print("\n=======================================================")
        print("               ASSISTANT LOOPS CLOSED")
        print("=======================================================")


if __name__ == "__main__":
    main()
