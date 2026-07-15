"""
WAV Playback Service for ZovaAI.
Reads WAV audio files locally and plays them back using sounddevice.
"""

from pathlib import Path
from typing import Optional
import sounddevice as sd  # type: ignore[import-untyped]
from scipy.io import wavfile  # type: ignore[import-untyped]

from src.core.logger import get_logger
from src.core.exceptions import AudioError

logger = get_logger("audio_playback")


def play_wav(wav_path: Path, device_index: Optional[int] = None) -> None:
    """
    Plays a WAV audio file locally using sounddevice.
    
    Args:
        wav_path: Absolute Path to the WAV file.
        device_index: Optional index of output audio device.
        
    Raises:
        AudioError: If file is missing, corrupted, or playback hardware fails.
    """
    if not wav_path.exists():
        raise AudioError(f"Audio file to play does not exist: {wav_path}")
        
    try:
        logger.info("Playing audio file: %s (device index: %s)", wav_path.name, device_index)
        
        # Read the WAV file using SciPy
        samplerate, data = wavfile.read(wav_path)
        
        # Play the audio and block execution until it finishes
        sd.play(data, samplerate, device=device_index)
        sd.wait()
        
        logger.debug("Playback finished successfully: %s", wav_path.name)
        
    except Exception as e:
        # Stop any active playback if an error occurs
        try:
            sd.stop()
        # pylint: disable=broad-exception-caught
        except Exception:
            pass
        raise AudioError(f"Failed to play audio file '{wav_path.name}': {str(e)}") from e
