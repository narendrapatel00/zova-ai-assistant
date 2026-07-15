"""
Unit tests for ZovaAI Audio Subsystem.
Tests microphone listing, stream initialization, VAD RMS silence detection,
WAV saving, and playback services using mocks.
"""

import wave
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest
import sounddevice as sd  # type: ignore[import-untyped]

from src.core.config import Config
from src.core.exceptions import AudioError
from src.audio.recorder import SounddeviceAudioRecorder
from src.audio.playback import play_wav


@pytest.fixture
def mock_config(tmp_path):
    """Fixture to generate a mock Config with test audio settings."""
    config = MagicMock(spec=Config)
    config.project_root = tmp_path
    
    config.audio = MagicMock()
    config.audio.sample_rate = 16000
    config.audio.channels = 1
    config.audio.chunk_size = 1280
    config.audio.device_index = 0
    config.audio.silence_threshold = 0.03
    config.audio.silence_seconds = 0.2  # Small timeout for fast testing
    
    return config


@pytest.fixture
def mock_sd_devices():
    """Fixture to mock sounddevice device query lists."""
    devices = [
        {
            "name": "Mock Input Mic",
            "max_input_channels": 1,
            "max_output_channels": 0,
            "default_samplerate": 16000.0
        },
        {
            "name": "Mock Output Speakers",
            "max_input_channels": 0,
            "max_output_channels": 2,
            "default_samplerate": 44100.0
        }
    ]
    with patch("sounddevice.query_devices", return_value=devices):
        with patch("sounddevice.default.device", new=[0, 1]):
            yield devices


class MockInputStream:
    """Mock implementation of sounddevice.InputStream for testing."""
    def __init__(self, device, samplerate, channels, callback, blocksize, dtype):
        self.device = device
        self.samplerate = samplerate
        self.channels = channels
        self.callback = callback
        self.blocksize = blocksize
        self.dtype = dtype
        self.active = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.active = False


@pytest.fixture
def patch_input_stream():
    """Fixture to patch sounddevice.InputStream to return MockInputStream."""
    active_streams = []

    def create_mock_stream(*args, **kwargs):
        stream = MockInputStream(*args, **kwargs)
        active_streams.append(stream)
        return stream

    with patch("sounddevice.InputStream", side_effect=create_mock_stream):
        yield active_streams


def test_device_enumeration(mock_sd_devices):
    """Checks that list_devices returns only input-capable microphones."""
    input_devices = SounddeviceAudioRecorder.list_devices()
    assert len(input_devices) == 1
    assert input_devices[0]["name"] == "Mock Input Mic"
    assert input_devices[0]["index"] == 0


def test_invalid_device_throws_error(mock_sd_devices, mock_config):
    """Verifies initializing recorder on invalid device index raises AudioError."""
    # Set to invalid index 5
    mock_config.audio.device_index = 5
    
    with pytest.raises(AudioError) as exc_info:
        SounddeviceAudioRecorder(mock_config)
    assert "Invalid audio device index: 5" in str(exc_info.value)


def test_output_only_device_throws_error(mock_sd_devices, mock_config):
    """Verifies initializing recorder on output-only device index raises AudioError."""
    # Set to output speakers index 1
    mock_config.audio.device_index = 1
    
    with pytest.raises(AudioError) as exc_info:
        SounddeviceAudioRecorder(mock_config)
    assert "has no input channels" in str(exc_info.value)


def test_recording_and_chunk_queue(mock_sd_devices, mock_config, patch_input_stream):
    """Verifies audio chunks are queued and recorded in buffers successfully."""
    recorder = SounddeviceAudioRecorder(mock_config)
    assert len(patch_input_stream) == 1
    stream = patch_input_stream[0]
    
    # 1. Verify get_audio_chunk blocks and reads from callback data
    dummy_data = np.ones((recorder.chunk_size, 1), dtype=np.float32) * 0.1
    stream.callback(dummy_data, recorder.chunk_size, None, sd.CallbackFlags())
    
    chunk = recorder.get_audio_chunk()
    assert len(chunk) == recorder.chunk_size
    assert np.allclose(chunk, 0.1)

    # 2. Test recording lifecycle
    recorder.start_recording()
    assert recorder.is_recording()
    
    # Feed two chunks
    stream.callback(dummy_data, recorder.chunk_size, None, sd.CallbackFlags())
    stream.callback(dummy_data, recorder.chunk_size, None, sd.CallbackFlags())
    
    # Stop recording and check file
    wav_path = recorder.stop_recording()
    assert not recorder.is_recording()
    assert wav_path.exists()
    
    # Check WAV properties
    with wave.open(str(wav_path), "rb") as w:
        assert w.getnchannels() == recorder.channels
        assert w.getsampwidth() == 2  # 16-bit
        assert w.getframerate() == recorder.sample_rate
        # Frame count should be equal to the length of combined chunks (2 * chunk_size)
        assert w.getnframes() == recorder.chunk_size * 2
        
    recorder.close()


def test_vad_silence_auto_stop(mock_sd_devices, mock_config, patch_input_stream):
    """Checks that the RMS VAD detects speech, then silences, and auto-stops recording."""
    recorder = SounddeviceAudioRecorder(mock_config)
    stream = patch_input_stream[0]
    
    recorder.start_recording()
    assert recorder.is_recording()
    
    # 1. Feed low energy (silence) but speech has not started yet
    silent_data = np.zeros((recorder.chunk_size, 1), dtype=np.float32)
    for _ in range(5):
        stream.callback(silent_data, recorder.chunk_size, None, sd.CallbackFlags())
    
    # Should still be recording because speech hasn't started
    assert recorder.is_recording()
    assert not recorder._has_speech_started

    # 2. Feed high energy (speech starts)
    loud_data = np.ones((recorder.chunk_size, 1), dtype=np.float32) * 0.15
    stream.callback(loud_data, recorder.chunk_size, None, sd.CallbackFlags())
    assert recorder._has_speech_started
    assert recorder._silence_duration == 0.0

    # 3. Feed low energy (silence begins)
    # The config timeout is 0.2s. 
    # At 16000Hz, each chunk of 1280 frames represents 1280 / 16000 = 0.08s.
    # 3 chunks represent 0.24s which exceeds the 0.2s silence timeout threshold.
    stream.callback(silent_data, recorder.chunk_size, None, sd.CallbackFlags())
    assert recorder.is_recording()  # 0.08s, still recording
    
    stream.callback(silent_data, recorder.chunk_size, None, sd.CallbackFlags())
    assert recorder.is_recording()  # 0.16s, still recording
    
    stream.callback(silent_data, recorder.chunk_size, None, sd.CallbackFlags())
    
    # 0.24s silence - VAD should auto-stop the recording state
    assert not recorder.is_recording()
    
    # The WAV file can be successfully stopped and saved
    wav_path = recorder.stop_recording()
    assert wav_path.exists()
    
    recorder.close()


def test_playback_success(tmp_path):
    """Checks play_wav reads file and triggers sounddevice playback."""
    dummy_wav = tmp_path / "test.wav"
    
    # Write a tiny valid mono 16-bit 16kHz WAV file (1600 frames = 0.1s)
    samplerate = 16000
    data = np.zeros(1600, dtype=np.int16)
    
    import scipy.io.wavfile as wavfile  # type: ignore[import-untyped]
    wavfile.write(str(dummy_wav), samplerate, data)
    
    with patch("sounddevice.play") as mock_play:
        with patch("sounddevice.wait") as mock_wait:
            play_wav(dummy_wav, device_index=1)
            
            mock_play.assert_called_once()
            args, kwargs = mock_play.call_args
            # Verify correct data shape passed
            assert np.array_equal(args[0], data)
            assert args[1] == samplerate
            assert kwargs["device"] == 1
            mock_wait.assert_called_once()


def test_playback_missing_file_throws_error():
    """Checks play_wav raises AudioError if the WAV file path does not exist."""
    non_existent = Path("non_existent_audio_file.wav")
    with pytest.raises(AudioError) as exc_info:
        play_wav(non_existent)
    assert "Audio file to play does not exist" in str(exc_info.value)
