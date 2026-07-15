"""
Speech-to-Text (STT) Subsystem CLI Demo for ZovaAI.
Starts the microphone recorder, records until silence, exports the audio
as a 16-bit mono WAV, and transcribes the speech offline using local Whisper.cpp.
"""

# pylint: disable=broad-exception-caught,duplicate-code

import sys
import time

from src.core.logger import LoggerSetup, get_logger
from src.core.exceptions import AudioError, STTError
from src.main import ZovaApp
from src.interfaces.audio_recorder import AudioRecorder
from src.interfaces.speech_recognizer import SpeechRecognizer

logger = get_logger("demo_stt")


def main() -> None:
    """Runs the voice capture, VAD silence stop, and Whisper.cpp transcription demo."""
    print("\n=======================================================")
    print("          ZOVA AI SPEECH-TO-TEXT (STT) DEMO")
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

    # 2. Resolve recorder and speech recognizer
    try:
        recorder = app.container.resolve(AudioRecorder)
        recognizer = app.container.resolve(SpeechRecognizer)
    except Exception as e:
        print(f"FATAL: DI dependency resolution failed: {e}", file=sys.stderr)
        app.close()
        sys.exit(1)

    print("\nDefault microphone stream active.")
    print("Speak your voice command now...")
    print("Recording will stop automatically after 1.5s of silence...")
    print("Press Ctrl+C to abort.")

    try:
        # Start recording session
        recorder.start_recording()
        
        while recorder.is_recording():
            recorder.get_audio_chunk()
            print(".", end="", flush=True)

        print("\n\nSilence detected. Stopping recording...")
        wav_path = recorder.stop_recording()
        print(f"WAV Audio file saved: {wav_path}")

        # 3. Transcribe file using Whisper
        print("\nLoading audio into Whisper.cpp and transcribing...")
        # Add brief delay to avoid concurrent file locking conflicts
        time.sleep(0.5)
        
        start_time = time.time()
        text = recognizer.transcribe(wav_path)
        latency = time.time() - start_time
        
        print("\n-------------------------------------------------------")
        print(f"Transcription complete (Time taken: {latency:.2f}s)")
        print(f"Result: \"{text}\"")
        print("-------------------------------------------------------")

    except KeyboardInterrupt:
        print("\n\nDemo aborted by user.")
    except (AudioError, STTError) as err:
        print(f"\nERROR: Speech processing pipeline failed: {err.message}", file=sys.stderr)
    finally:
        app.close()
        print("\n=======================================================")
        print("                 STT DEMO COMPLETE")
        print("=======================================================")


if __name__ == "__main__":
    main()
