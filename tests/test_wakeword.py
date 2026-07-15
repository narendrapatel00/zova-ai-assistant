"""
Unit tests for ZovaAI Wake Word Subsystem.
Tests model loading, threshold parsing, cooldown timers, disabled states,
and exception recovery using mocks.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.core.config import Config
from src.core.exceptions import WakeWordError, AudioError
from src.interfaces.audio_recorder import AudioRecorder
from src.wakeword.engine import OpenWakeWordDetector
from src.wakeword.service import WakeWordListeningService


@pytest.fixture
def mock_config(tmp_path):
    """Fixture to generate a mock Config with wake word settings."""
    config = MagicMock(spec=Config)
    config.project_root = tmp_path
    
    # Audio settings
    config.audio = MagicMock()
    config.audio.sample_rate = 16000
    config.audio.channels = 1
    config.audio.chunk_size = 1280
    config.audio.device_index = 0
    config.audio.silence_threshold = 0.03
    config.audio.silence_seconds = 1.5
    
    # Wake Word settings
    config.wakeword = MagicMock()
    config.wakeword.model_path = ""
    config.wakeword.threshold = 0.6
    config.wakeword.cooldown_seconds = 0.2  # Small cooldown for fast testing
    config.wakeword.enabled = True
    config.wakeword.inference_interval_ms = 80
    
    return config


@pytest.fixture
def mock_oww_model():
    """Fixture to mock the Model class inside src.wakeword.engine."""
    with patch("src.wakeword.engine.Model") as mock:
        instance = MagicMock()
        instance.predict.return_value = {"hey_jarvis": 0.0}
        mock.return_value = instance
        yield mock, instance


def test_model_loading_default(mock_config, mock_oww_model):
    """Checks that detector initializes with default model name when path is empty."""
    mock_class, mock_inst = mock_oww_model
    detector = OpenWakeWordDetector(mock_config)
    
    mock_class.assert_called_once_with(wakeword_models=["hey_jarvis"])
    assert detector.get_wake_word_name() == "hey_jarvis"


def test_model_loading_custom(mock_config, mock_oww_model, tmp_path):
    """Checks that detector resolves and loads custom ONNX paths when configured."""
    mock_class, mock_inst = mock_oww_model
    custom_model = tmp_path / "custom_wake.onnx"
    custom_model.write_text("dummy model content")
    
    # Mock resolved path
    mock_config.resolve_path.return_value = custom_model
    mock_config.wakeword.model_path = "models/custom_wake.onnx"
    
    detector = OpenWakeWordDetector(mock_config)
    
    mock_class.assert_called_once_with(wakeword_models=[str(custom_model)])
    assert detector.get_wake_word_name() == "custom_wake"


def test_model_loading_missing_throws_error(mock_config, tmp_path):
    """Checks that detector raises WakeWordError if custom model file does not exist."""
    mock_config.resolve_path.return_value = tmp_path / "missing.onnx"
    mock_config.wakeword.model_path = "models/missing.onnx"
    
    with pytest.raises(WakeWordError) as exc_info:
        OpenWakeWordDetector(mock_config)
    assert "Custom wake-word model path not found" in str(exc_info.value)


def test_threshold_logic(mock_config, mock_oww_model):
    """Checks that detect returns True only when score meets or exceeds threshold."""
    detector = OpenWakeWordDetector(mock_config)
    _, mock_inst = mock_oww_model
    chunk = np.zeros(1280, dtype=np.float32)

    # 1. Under threshold
    mock_inst.predict.return_value = {"hey_jarvis": 0.3}
    assert not detector.detect(chunk)

    # 2. Equal/Over threshold
    mock_inst.predict.return_value = {"hey_jarvis": 0.65}
    assert detector.detect(chunk)


def test_service_disabled_mode(mock_config):
    """Checks that start does not run listening threads when enabled: False."""
    mock_config.wakeword.enabled = False
    recorder = MagicMock(spec=AudioRecorder)
    detector = MagicMock(spec=OpenWakeWordDetector)
    
    service = WakeWordListeningService(mock_config, recorder, detector)
    service.start()
    
    assert not service.is_running()
    recorder.subscribe.assert_not_called()


@pytest.fixture
def patch_play_wav():
    """Fixture to mock play_wav so no beep plays during tests."""
    with patch("src.wakeword.service.play_wav") as mock:
        yield mock


def test_service_cooldown_and_callback(mock_config, patch_play_wav):
    """Checks that callback is fired upon detection and cooldown silences immediate triggers."""
    recorder = MagicMock(spec=AudioRecorder)
    recorder.is_recording.return_value = False

    detector = MagicMock(spec=OpenWakeWordDetector)
    detector.get_wake_word_name.return_value = "hey_jarvis"
    detector.detect.side_effect = [True, True]

    service = WakeWordListeningService(mock_config, recorder, detector)
    
    callback_fired = 0
    def test_cb():
        nonlocal callback_fired
        callback_fired += 1
        
    service.register_callback(test_cb)
    
    # 1. Start the service (mocks subscribe)
    service.start()
    assert service.is_running()
    recorder.subscribe.assert_called_once_with(service)
    
    # 2. Feed 1st chunk via observer pattern (VAD detect: True -> trigger cb)
    service.on_audio_chunk(np.zeros(1280, dtype=np.float32))
    
    # Wait up to 1s for background thread processing
    start_t = time.time()
    while callback_fired == 0 and time.time() - start_t < 1.0:
        time.sleep(0.01)
    assert callback_fired == 1
    
    # 3. Feed 2nd chunk immediately (VAD detect: True -> hits cooldown -> no cb trigger)
    service.on_audio_chunk(np.zeros(1280, dtype=np.float32))
    time.sleep(0.1)
    assert callback_fired == 1
    
    # 4. Stop service (mocks unsubscribe)
    service.stop()
    assert not service.is_running()
    recorder.unsubscribe.assert_called_once_with(service)


def test_graceful_stream_error_recovery(mock_config, patch_play_wav):
    """Checks that the listening thread logs and retries when queue processing triggers errors."""
    recorder = MagicMock(spec=AudioRecorder)
    recorder.is_recording.return_value = False
    
    detector = MagicMock(spec=OpenWakeWordDetector)
    detector.get_wake_word_name.return_value = "hey_jarvis"
    detector.detect.side_effect = [WakeWordError("Inference crashed"), False]
    
    service = WakeWordListeningService(mock_config, recorder, detector)
    service.start()
    
    # Feed two chunks
    service.on_audio_chunk(np.zeros(1280, dtype=np.float32))
    service.on_audio_chunk(np.zeros(1280, dtype=np.float32))
    
    # Wait for background thread to run detect
    start_t = time.time()
    while detector.detect.call_count < 2 and time.time() - start_t < 1.0:
        time.sleep(0.01)
        
    service.stop()
    
    assert detector.detect.call_count == 2
