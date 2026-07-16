"""
Text-to-Speech (TTS) Subsystem CLI Demo for ZovaAI.
Resolves the SpeechSynthesisService from the DI container and synthesizes
input text into immediate voice output.
"""

# pylint: disable=broad-exception-caught

import sys

from src.core.logger import LoggerSetup, get_logger
from src.core.exceptions import TTSError
from src.main import ZovaApp
from src.tts.service import SpeechSynthesisService

logger = get_logger("demo_tts")


def main() -> None:
    """Runs the offline Piper text-to-speech synthesis and voice playback demo."""
    print("\n=======================================================")
    print("          ZOVA AI TEXT-TO-SPEECH (TTS) DEMO")
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

    # 2. Resolve speech synthesis service
    try:
        tts_service = app.container.resolve(SpeechSynthesisService)
    except Exception as e:
        print(f"FATAL: DI dependency resolution failed: {e}", file=sys.stderr)
        app.close()
        sys.exit(1)

    # Ask user for input phrase or use default
    default_text = "Welcome to Zova, your offline-first personal AI assistant. I am ready."
    print(f"\nEnter text to synthesize (Press Enter to use default: '{default_text}'):")
    user_input = input("> ").strip()

    speak_text = user_input if user_input else default_text

    print(f"\nSynthesizing text: \"{speak_text}\"")
    print("Please wait...")

    try:
        # Run speech playback
        tts_service.speak(speak_text)
        print("Vocalization finished.")

    except TTSError as err:
        print(f"\nERROR: Speech synthesis failed: {err.message}", file=sys.stderr)
    finally:
        app.close()
        print("\n=======================================================")
        print("                 TTS DEMO COMPLETE")
        print("=======================================================")


if __name__ == "__main__":
    main()
