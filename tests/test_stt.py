"""
Unit tests for ZovaAI Speech-to-Text (STT) Subsystem.
Tests model loading, WAV structure validation (empty frames, sample rates, channels),
transcription segment parsing, and exception wrapping using mocks.
"""

import wave
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.core.config import Config
from src.core.exceptions import STTError
from src.interfaces.speech_recognizer import SpeechRecognizer
from src.stt.recognizer import WhisperSpeechRecognizer
from src.stt.service import SpeechRecognitionService


@pytest.fixture
def mock_config(tmp_path):
    """Fixture to generate a mock Config with STT settings."""
    config = MagicMock(spec=Config)
    config.project_root = tmp_path

    config.stt = MagicMock()
    config.stt.enabled = True
    config.stt.model_path = Path("models/whisper/ggml-base.en.bin")
    config.stt.language = "en"
    config.stt.threads = 4
    config.stt.translate = False
    config.stt.beam_size = 5
    config.stt.temperature = 0.0

    # Mock resolve_path to return a valid dummy path
    config.resolve_path.side_effect = lambda p: tmp_path / Path(p).name

    return config


@pytest.fixture
def mock_whisper_model():
    """Fixture to mock the Model class inside src.stt.recognizer."""
    with patch("src.stt.recognizer.Model") as mock:
        instance = MagicMock()
        mock.return_value = instance
        yield mock, instance


def create_dummy_wav(file_path: Path, sample_rate: int = 16000, channels: int = 1, frames: int = 1600) -> None:
    """Helper method to write a dummy WAV file for test validation."""
    with wave.open(str(file_path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        # Write quiet dummy bytes
        w.writeframes(b"\x00" * (frames * 2 * channels))


def test_model_loading_success(mock_config, mock_whisper_model, tmp_path):
    """Checks that recognizer initializes and resolves model path successfully."""
    mock_class, _ = mock_whisper_model
    model_file = tmp_path / "ggml-base.en.bin"
    model_file.write_text("dummy model binary")

    recognizer = WhisperSpeechRecognizer(mock_config)

    mock_class.assert_called_once_with(
        str(model_file),
        n_threads=4,
        redirect_whispercpp_logs_to=False
    )
    assert recognizer.enabled is True


def test_model_loading_missing_throws_error(mock_config):
    """Checks that STTError is raised if the Whisper binary model file is missing."""
    # Do not write model binary to tmp_path
    with pytest.raises(STTError) as exc_info:
        WhisperSpeechRecognizer(mock_config)
    assert "Whisper model file not found" in str(exc_info.value)


def test_transcribe_success(mock_config, mock_whisper_model, tmp_path):
    """Checks successful WAV transcription text assembly from segment mocks."""
    _, mock_inst = mock_whisper_model

    # Mock model file existence
    model_file = tmp_path / "ggml-base.en.bin"
    model_file.write_text("model")

    # Mock output segments
    seg1 = MagicMock()
    seg1.text = "Hello "
    seg2 = MagicMock()
    seg2.text = "Jarvis!"
    mock_inst.transcribe.return_value = [seg1, seg2]

    recognizer = WhisperSpeechRecognizer(mock_config)

    # Create valid dummy WAV
    wav_file = tmp_path / "command.wav"
    create_dummy_wav(wav_file, sample_rate=16000, channels=1, frames=3200)

    # Call transcribe
    text = recognizer.transcribe(wav_file)

    assert text == "Hello Jarvis!"
    mock_inst.transcribe.assert_called_once_with(
        str(wav_file),
        language="en",
        beam_size=5,
        temperature=0.0
    )


def test_transcribe_disabled_mode_throws_error(mock_config, mock_whisper_model, tmp_path):
    """Checks that calling transcribe when enabled is False raises STTError."""
    # Write model file to bypass loader checks
    model_file = tmp_path / "ggml-base.en.bin"
    model_file.write_text("model")

    mock_config.stt.enabled = False
    recognizer = WhisperSpeechRecognizer(mock_config)

    wav_file = tmp_path / "test.wav"

    with pytest.raises(STTError) as exc_info:
        recognizer.transcribe(wav_file)
    assert "SpeechRecognizer is disabled" in str(exc_info.value)


def test_transcribe_missing_file_throws_error(mock_config, mock_whisper_model, tmp_path):
    """Checks that transcribing a non-existent file raises STTError."""
    model_file = tmp_path / "ggml-base.en.bin"
    model_file.write_text("model")

    recognizer = WhisperSpeechRecognizer(mock_config)
    missing_file = tmp_path / "missing.wav"

    with pytest.raises(STTError) as exc_info:
        recognizer.transcribe(missing_file)
    assert "Audio file to transcribe not found" in str(exc_info.value)


def test_transcribe_invalid_samplerate_throws_error(mock_config, mock_whisper_model, tmp_path):
    """Checks that a WAV file with non-16kHz sample rate causes an STTError."""
    model_file = tmp_path / "ggml-base.en.bin"
    model_file.write_text("model")

    recognizer = WhisperSpeechRecognizer(mock_config)

    # Create invalid 44.1kHz WAV
    wav_file = tmp_path / "invalid_rate.wav"
    create_dummy_wav(wav_file, sample_rate=44100, channels=1)

    with pytest.raises(STTError) as exc_info:
        recognizer.transcribe(wav_file)
    assert "Unsupported sample rate" in str(exc_info.value)


def test_transcribe_invalid_channels_throws_error(mock_config, mock_whisper_model, tmp_path):
    """Checks that a stereo (2 channels) WAV file causes an STTError."""
    model_file = tmp_path / "ggml-base.en.bin"
    model_file.write_text("model")

    recognizer = WhisperSpeechRecognizer(mock_config)

    # Create stereo 16kHz WAV
    wav_file = tmp_path / "stereo.wav"
    create_dummy_wav(wav_file, sample_rate=16000, channels=2)

    with pytest.raises(STTError) as exc_info:
        recognizer.transcribe(wav_file)
    assert "Unsupported channels count" in str(exc_info.value)


def test_transcribe_empty_file_throws_error(mock_config, mock_whisper_model, tmp_path):
    """Checks that an empty WAV file containing 0 frames causes an STTError."""
    model_file = tmp_path / "ggml-base.en.bin"
    model_file.write_text("model")

    recognizer = WhisperSpeechRecognizer(mock_config)

    # Create empty WAV
    wav_file = tmp_path / "empty.wav"
    create_dummy_wav(wav_file, sample_rate=16000, channels=1, frames=0)

    with pytest.raises(STTError) as exc_info:
        recognizer.transcribe(wav_file)
    assert "Empty recording" in str(exc_info.value)


def test_transcribe_exception_wrapping(mock_config, mock_whisper_model, tmp_path):
    """Checks that general transcription exceptions are caught and wrapped in STTError."""
    _, mock_inst = mock_whisper_model
    model_file = tmp_path / "ggml-base.en.bin"
    model_file.write_text("model")

    # Make transcribe raise a runtime error
    mock_inst.transcribe.side_effect = RuntimeError("Whisper.cpp core dump")

    recognizer = WhisperSpeechRecognizer(mock_config)

    wav_file = tmp_path / "valid.wav"
    create_dummy_wav(wav_file, sample_rate=16000, channels=1, frames=1600)

    with pytest.raises(STTError) as exc_info:
        recognizer.transcribe(wav_file)
    assert "Whisper transcription failed" in str(exc_info.value)


def test_service_transcribe_delegation():
    """Checks that SpeechRecognitionService forwards transcribe calls to recognizer."""
    mock_recognizer = MagicMock(spec=SpeechRecognizer)
    mock_recognizer.transcribe.return_value = "transcribed voice text"

    service = SpeechRecognitionService(mock_recognizer)

    dummy_path = Path("fake_audio.wav")
    text = service.transcribe_audio(dummy_path)

    assert text == "transcribed voice text"
    mock_recognizer.transcribe.assert_called_once_with(dummy_path)
