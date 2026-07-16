"""
Unit tests for ZovaAI Text-to-Speech (TTS) Subsystem.
Tests path validations, subprocess execution parameters, output validation,
WAV playback delegation, cleanup of temporary files, and service wrappers using mocks.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.core.config import Config
from src.core.exceptions import TTSError, AudioError
from src.interfaces.speech_synthesizer import SpeechSynthesizer
from src.tts.synthesizer import PiperSpeechSynthesizer
from src.tts.service import SpeechSynthesisService


@pytest.fixture
def mock_config(tmp_path):
    """Fixture to generate a mock Config with TTS settings."""
    config = MagicMock(spec=Config)
    config.project_root = tmp_path

    config.audio = MagicMock()
    config.audio.device_index = 0

    config.tts = MagicMock()
    config.tts.enabled = True
    config.tts.executable_path = tmp_path / "bin" / "piper" / "piper.exe"
    config.tts.model_path = tmp_path / "models" / "piper" / "voice.onnx"
    config.tts.config_path = tmp_path / "models" / "piper" / "voice.onnx.json"
    config.tts.output_dir = tmp_path / "temp" / "tts"

    return config


def test_resource_verification_success(mock_config):
    """Checks that synthesizer initializes successfully if files exist."""
    # Write mock files
    mock_config.tts.executable_path.parent.mkdir(parents=True, exist_ok=True)
    mock_config.tts.executable_path.write_text("bin")

    mock_config.tts.model_path.parent.mkdir(parents=True, exist_ok=True)
    mock_config.tts.model_path.write_text("model")
    mock_config.tts.config_path.write_text("config")

    synthesizer = PiperSpeechSynthesizer(mock_config)
    assert synthesizer.enabled is True


def test_resource_verification_missing_exe_throws_error(mock_config):
    """Checks that TTSError is raised if the Piper binary is missing."""
    # Do not write executable path
    mock_config.tts.model_path.parent.mkdir(parents=True, exist_ok=True)
    mock_config.tts.model_path.write_text("model")
    mock_config.tts.config_path.write_text("config")

    with pytest.raises(TTSError) as exc_info:
        PiperSpeechSynthesizer(mock_config)
    assert "Piper executable not found" in str(exc_info.value)


def test_resource_verification_missing_model_throws_error(mock_config):
    """Checks that TTSError is raised if the voice ONNX model file is missing."""
    # Write exe but do not write model paths
    mock_config.tts.executable_path.parent.mkdir(parents=True, exist_ok=True)
    mock_config.tts.executable_path.write_text("bin")
    mock_config.tts.config_path.parent.mkdir(parents=True, exist_ok=True)
    mock_config.tts.config_path.write_text("config")

    with pytest.raises(TTSError) as exc_info:
        PiperSpeechSynthesizer(mock_config)
    assert "Piper ONNX model not found" in str(exc_info.value)


@pytest.fixture
def mock_synthesizer(mock_config):
    """Fixture to provide a fully initialized synthesizer with mocked file exists checks."""
    mock_config.tts.executable_path.parent.mkdir(parents=True, exist_ok=True)
    mock_config.tts.executable_path.write_text("bin")
    mock_config.tts.model_path.parent.mkdir(parents=True, exist_ok=True)
    mock_config.tts.model_path.write_text("model")
    mock_config.tts.config_path.write_text("config")

    return PiperSpeechSynthesizer(mock_config)


def test_synthesize_success(mock_synthesizer, mock_config, tmp_path):
    """Checks successful Piper subprocess invocation and file output validations."""
    output_wav = tmp_path / "output.wav"
    input_text = "Testing offline synthesizer."

    # Mock subprocess run to simulate successful synthesis and write output wav
    def mock_run(cmd, input, capture_output, check):
        # Create output file to pass exists checks
        output_wav.write_text("mock wav bytes")
        mock_res = MagicMock()
        mock_res.returncode = 0
        return mock_res

    with patch("subprocess.run", side_effect=mock_run) as mock_subprocess_run:
        result_path = mock_synthesizer.synthesize(input_text, output_wav)

        assert result_path == output_wav
        assert output_wav.exists()

        # Verify correct args passed to subprocess
        mock_subprocess_run.assert_called_once()
        args, kwargs = mock_subprocess_run.call_args
        assert args[0][0] == str(mock_config.tts.executable_path)
        assert "--model" in args[0]
        assert str(mock_config.tts.model_path) in args[0]
        assert str(output_wav) in args[0]
        assert kwargs["input"] == input_text.encode("utf-8")


def test_synthesize_empty_input_throws_error(mock_synthesizer, tmp_path):
    """Checks that passing empty text to synthesize raises a TTSError."""
    output_wav = tmp_path / "output.wav"
    with pytest.raises(TTSError) as exc_info:
        mock_synthesizer.synthesize("   ", output_wav)
    assert "text input cannot be empty" in str(exc_info.value)


def test_synthesize_process_failure_throws_error(mock_synthesizer, tmp_path):
    """Checks that a non-zero exit code in the Piper subprocess raises a TTSError."""
    output_wav = tmp_path / "output.wav"

    with patch("subprocess.run") as mock_run:
        mock_res = MagicMock()
        mock_res.returncode = 1
        mock_res.stderr = b"Out of memory error in ONNX runtime."
        mock_run.return_value = mock_res

        with pytest.raises(TTSError) as exc_info:
            mock_synthesizer.synthesize("Hello", output_wav)
        assert "Piper synthesis process exited with code 1" in str(exc_info.value)


def test_speak_immediate_playback_and_cleanup(mock_synthesizer):
    """Checks that speak generates wav, plays it, and deletes the temporary file."""
    text = "Speak aloud immediately."

    with patch.object(Path, "exists", return_value=True):
        with patch.object(mock_synthesizer, "synthesize") as mock_synth_method:
            with patch("src.tts.synthesizer.play_wav") as mock_play_wav:
                with patch("src.tts.synthesizer.os.unlink") as mock_unlink:

                    # Make synthesize return the output path passed to it
                    mock_synth_method.side_effect = lambda t, p: p

                    mock_synthesizer.speak(text)

                    # Retrieve the path dynamically generated by speak()
                    called_path = mock_synth_method.call_args[0][1]

                    mock_synth_method.assert_called_once_with(text, called_path)
                    mock_play_wav.assert_called_once_with(called_path, device_index=mock_synthesizer.config.audio.device_index)
                    mock_unlink.assert_called_once_with(called_path)


def test_speak_playback_failure_raises_error(mock_synthesizer):
    """Checks that playback failure (AudioError) is wrapped and raised as TTSError."""
    text = "Speak aloud failure test."

    with patch.object(Path, "exists", return_value=True):
        with patch.object(mock_synthesizer, "synthesize") as mock_synth_method:
            with patch("src.tts.synthesizer.play_wav", side_effect=AudioError("Hardware lock error")):
                with patch("src.tts.synthesizer.os.unlink") as mock_unlink:

                    mock_synth_method.side_effect = lambda t, p: p

                    with pytest.raises(TTSError) as exc_info:
                        mock_synthesizer.speak(text)
                    assert "Failed to play back synthesized speech" in str(exc_info.value)

                    called_path = mock_synth_method.call_args[0][1]
                    mock_unlink.assert_called_once_with(called_path)


def test_service_speak_delegation():
    """Checks that SpeechSynthesisService forwards synthesis calls to the recognizer."""
    mock_synth = MagicMock(spec=SpeechSynthesizer)
    service = SpeechSynthesisService(mock_synth)

    # 1. Test speak
    service.speak("Test text")
    mock_synth.speak.assert_called_once_with("Test text")

    # 2. Test synthesize to file
    dummy_dest = Path("destination.wav")
    service.synthesize_to_file("Text to write", dummy_dest)
    mock_synth.synthesize.assert_called_once_with("Text to write", dummy_dest)
