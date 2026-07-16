"""
Wake Word Subsystem CLI Demo for ZovaAI.
Starts the background wake-word service, listens for "Jarvis",
plays a local activation beep on success, captures your voice command
until silence is detected, and saves it as a local WAV file.
"""

# pylint: disable=broad-exception-caught,too-many-statements,duplicate-code

import sys
import time
import threading

from src.core.logger import LoggerSetup, get_logger
from src.core.exceptions import AudioError, WakeWordError
from src.main import ZovaApp
from src.interfaces.audio_recorder import AudioRecorder
from src.wakeword.interfaces import WakeWordService

logger = get_logger("demo_wakeword")


def main() -> None:
    """Runs the wake-word listener, VAD recorder, and WAV save sequence."""
    print("\n=======================================================")
    print("           ZOVA AI WAKE WORD SUBSYSTEM DEMO")
    print("=======================================================")

    # 1. Initialize Zova App and register DI services
    app = ZovaApp()
    try:
        app.initialize()
        # Set console logs to INFO for clean readability
        LoggerSetup.initialize(log_level="INFO")
    except Exception as e:
        print(f"FATAL: Application bootstrap failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Resolve audio and wake-word components
    try:
        recorder = app.container.resolve(AudioRecorder)
        ww_service = app.container.resolve(WakeWordService)
    except Exception as e:
        print(f"FATAL: Failed to resolve DI dependencies: {e}", file=sys.stderr)
        app.close()
        sys.exit(1)

    # 3. Create a thread event to synchronize wake word triggers
    ww_detected_event = threading.Event()

    def handle_wake_word() -> None:
        """Callback handler executed when Jarvis is detected."""
        ww_detected_event.set()

    # Register callback
    # The service will play the confirmation sound automatically, then fire this.
    ww_service.register_callback(handle_wake_word)

    print("\n[Step 1/3] Starting Wake-Word listening thread...")
    try:
        ww_service.start()
    except (WakeWordError, AudioError) as err:
        print(f"ERROR: Wake-word engine failed to start: {err.message}", file=sys.stderr)
        app.close()
        sys.exit(1)

    print("\nListening for 'Jarvis'...")
    print("Say 'Jarvis' or 'Hey Jarvis' now to activate.")
    print("Press Ctrl+C to abort.")

    try:
        # Wait for wake word detection event (with short timeout to check for interrupts)
        while not ww_detected_event.is_set():
            time.sleep(0.1)

        # 4. Begin audio command recording
        print("\n[Step 2/3] Wake word detected! Beginning voice command capture...")
        print("Speak your command now...")
        print("Recording will stop automatically after 1.5s of silence...")

        recorder.start_recording()

        # Read chunks from the stream to keep the recorder callback feeding VAD
        while recorder.is_recording():
            recorder.get_audio_chunk()
            print(".", end="", flush=True)

        print("\n\n[Step 3/3] Silence detected. Stopping recording...")
        wav_path = recorder.stop_recording()
        print(f"\nCommand captured successfully: {wav_path}")
        print("Wake-word trigger cycle complete.")

    except KeyboardInterrupt:
        print("\n\nDemo aborted by user.")
    except Exception as e:
        print(f"\nUnexpected error during demo: {e}", file=sys.stderr)
    finally:
        app.close()
        print("\n=======================================================")
        print("               WAKE WORD DEMO COMPLETE")
        print("=======================================================")


if __name__ == "__main__":
    main()
