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
from src.assistant.events import AssistantEventObserver

logger = get_logger("demo_assistant")


class ConsoleAssistantObserver(AssistantEventObserver):
    """Event observer that logs state transitions directly to the console."""

    def on_waiting_for_wake_word(self) -> None:
        print("\n-------------------------------------------------------")
        print(" [Idle] Ready. Listening for wake word: 'Jarvis'...")
        print("-------------------------------------------------------")

    def on_wake_word_detected(self) -> None:
        print("\n>>> [Wake Word Detected] Jarvis matched! <<<")

    def on_recording_started(self) -> None:
        print("[Record] Microphone active. Speak your voice command now...")

    def on_recording_finished(self, wav_path) -> None:
        print(f"[Record] Sound captured. Exported WAV file: {wav_path.name}")

    def on_transcription_finished(self, text: str) -> None:
        print(f"[STT] Transcribed text: \"{text}\"")

    def on_llm_start(self, prompt: str) -> None:
        print("[LLM] Dispatched query to local Ollama server...")

    def on_llm_response(self, response: str) -> None:
        print(f"[LLM] Brain response: \"{response}\"")

    def on_tts_start(self, text: str) -> None:
        print("[TTS] Running offline speech synthesis...")

    def on_tts_finished(self) -> None:
        print("[TTS] Vocalization playback complete.")

    def on_error(self, error_message: str) -> None:
        print(f"\n[ERROR] Pipeline exception encountered: {error_message}")


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

    # 2. Resolve orchestrator from DI Container
    try:
        orchestrator = app.container.resolve(AssistantOrchestrator)
    except Exception as e:
        print(f"FATAL: DI dependency resolution failed: {e}", file=sys.stderr)
        app.close()
        sys.exit(1)

    # 3. Instantiate and register the console events observer
    observer = ConsoleAssistantObserver()
    orchestrator.register_observer(observer)

    print("\nStarting orchestrator services...")
    print("Press Ctrl+C to stop the assistant and shutdown.")

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
