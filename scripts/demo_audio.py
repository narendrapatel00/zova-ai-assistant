"""
Audio Subsystem CLI Demo for ZovaAI.
Lists microphones, records audio using VAD auto-stop, saves to a WAV file,
and plays it back to verify both recording and playback subsystems are functional.
"""

import sys
import time

from src.core.config import Config
from src.core.logger import LoggerSetup, get_logger
from src.core.exceptions import AudioError
from src.audio.recorder import SounddeviceAudioRecorder
from src.audio.playback import play_wav

logger = get_logger("demo_audio")


def main() -> None:
    """Executes the microphone list, record, VAD detect, save, and playback demo."""
    # 1. Load config and initialize logger to console only (clean interface)
    config = Config()
    LoggerSetup.initialize(log_level="INFO")

    print("\n=======================================================")
    print("           ZOVA AI AUDIO SUBSYSTEM DEMO")
    print("=======================================================")

    # 2. List available microphones
    print("\n[Step 1/4] Querying Input Audio Devices...")
    try:
        devices = SounddeviceAudioRecorder.list_devices()
        if not devices:
            print("ERROR: No input audio devices (microphones) discovered.")
            sys.exit(1)
        for dev in devices:
            print(
                f"  🎤 Index {dev['index']}: {dev['name']} "
                f"(Channels: {dev['max_input_channels']}, "
                f"Sample Rate: {dev['default_samplerate']}Hz)"
            )
    except AudioError as ae:
        print(f"ERROR: Device listing failed: {ae.message}", file=sys.stderr)
        sys.exit(1)

    # 3. Start audio stream
    print("\n[Step 2/4] Initializing Default Microphone Stream...")
    try:
        recorder = SounddeviceAudioRecorder(config)
    except AudioError as ae:
        print(f"ERROR: Stream initialization failed: {ae.message}", file=sys.stderr)
        print("Please verify that a working microphone is connected to this PC.")
        sys.exit(1)

    print("\nInputStream Active. Speak into your microphone...")
    print(f"Recording starts when you speak (threshold: {recorder.silence_threshold}).")
    print(f"Recording stops automatically after {recorder.silence_seconds}s of silence.")
    print("Press Ctrl+C to abort.")

    try:
        recorder.start_recording()

        # Read chunks. get_audio_chunk() blocks for blocksize (80ms), regulating the loop.
        while recorder.is_recording():
            recorder.get_audio_chunk()
            # print a subtle dot to show active capture
            print(".", end="", flush=True)

        print("\n\n[Step 3/4] Silence detected. Stopping recording...")
        wav_path = recorder.stop_recording()
        print(f"WAV Audio file saved: {wav_path}")

        # 4. Playback audio
        print("\n[Step 4/4] Playing back the recorded audio track...")
        # Add a small sleep to avoid playback audio clicking immediately
        time.sleep(0.5)
        play_wav(wav_path, device_index=config.audio.device_index)
        print("Playback finished.")

    except KeyboardInterrupt:
        print("\n\nDemo aborted by user.")
    except AudioError as ae:
        print(f"\nERROR: Audio operation failed: {ae.message}", file=sys.stderr)
    finally:
        recorder.close()
        print("\n=======================================================")
        print("                 AUDIO DEMO COMPLETE")
        print("=======================================================")


if __name__ == "__main__":
    main()
