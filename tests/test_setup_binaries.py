"""
Unit tests for ZovaAI BinarySetupManager.
Tests downloading, zip extraction, directories ensuring, and resume logic using mocks.
"""

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.core.config import Config
from src.core.exceptions import SetupError
from scripts.setup_binaries import BinarySetupManager


@pytest.fixture
def temp_dirs(tmp_path):
    """Fixture to provide temporary paths representing ZovaAI directories."""
    paths = {
        "project_root": tmp_path,
        "whisper_dir": tmp_path / "models" / "whisper",
        "piper_model_dir": tmp_path / "models" / "piper",
        "piper_bin_dir": tmp_path / "bin" / "piper",
        "temp_tts_dir": tmp_path / "temp" / "tts",
        "temp_dir": tmp_path / "temp"
    }
    return paths


@pytest.fixture
def mock_config(temp_dirs):
    """Fixture to provide a mocked Config instance containing temporary paths."""
    config = MagicMock(spec=Config)
    config.project_root = temp_dirs["project_root"]
    
    # Mock Speech Recognition Configuration
    config.stt = MagicMock()
    config.stt.model_path = temp_dirs["whisper_dir"] / "ggml-base.en.bin"
    
    # Mock Speech Synthesis Configuration
    config.tts = MagicMock()
    config.tts.model_path = temp_dirs["piper_model_dir"] / "voice.onnx"
    config.tts.config_path = temp_dirs["piper_model_dir"] / "voice.onnx.json"
    config.tts.executable_path = temp_dirs["piper_bin_dir"] / "piper.exe"
    config.tts.output_dir = temp_dirs["temp_tts_dir"]
    
    # Mock Setup Configuration URLs
    config.setup = MagicMock()
    config.setup.piper_zip_url = "http://example.com/piper.zip"
    config.setup.piper_voice_url = "http://example.com/voice.onnx"
    config.setup.piper_voice_config_url = "http://example.com/voice.onnx.json"
    config.setup.whisper_model_url = "http://example.com/whisper.bin"
    
    return config


def test_ensure_directories(mock_config, temp_dirs):
    """Checks that BinarySetupManager successfully creates missing target folders."""
    manager = BinarySetupManager(mock_config)
    
    # Verify directories do not exist yet
    for d in temp_dirs.values():
        if d != temp_dirs["project_root"]:
            assert not d.exists()
            
    manager.ensure_directories()
    
    # Verify directories are successfully created
    for d in temp_dirs.values():
        assert d.exists()


def test_download_file_skips_if_complete(mock_config, tmp_path):
    """Verifies download is skipped if a file with matching expected size already exists."""
    manager = BinarySetupManager(mock_config)
    dest_file = tmp_path / "existing.bin"
    content = b"dummy content data"
    
    # Write local file
    dest_file.write_bytes(content)
    
    # Mock HTTP response headers with matching size
    mock_head_res = MagicMock()
    mock_head_res.headers = {
        "content-length": str(len(content)),
        "accept-ranges": "bytes"
    }
    
    with patch("requests.head", return_value=mock_head_res) as mock_head:
        with patch("requests.get") as mock_get:
            manager.download_file("http://example.com/file.bin", dest_file, "Test Skip")
            
            # Verify head request was run, but get request was skipped
            mock_head.assert_called_once()
            mock_get.assert_not_called()
            
    assert dest_file.read_bytes() == content


def test_download_file_resets_if_local_larger(mock_config, tmp_path):
    """Verifies download file is deleted and restarted if local size exceeds expected size."""
    manager = BinarySetupManager(mock_config)
    dest_file = tmp_path / "oversized.bin"
    dest_file.write_bytes(b"very long oversized local content data")
    
    # Mock headers indicating smaller expected size
    mock_head_res = MagicMock()
    mock_head_res.headers = {
        "content-length": "10",
        "accept-ranges": "bytes"
    }
    
    # Mock GET response to return dummy download data
    mock_get_res = MagicMock()
    mock_get_res.status_code = 200
    mock_get_res.headers = {"content-length": "10"}
    mock_get_res.iter_content.return_value = [b"1234567890"]
    mock_get_res.__enter__.return_value = mock_get_res
    
    with patch("requests.head", return_value=mock_head_res):
        with patch("requests.get", return_value=mock_get_res):
            manager.download_file("http://example.com/file.bin", dest_file, "Test Reset")
            
    assert dest_file.read_bytes() == b"1234567890"


def test_download_file_resumes_if_partial(mock_config, tmp_path):
    """Verifies download can resume from local file size using Range headers."""
    manager = BinarySetupManager(mock_config)
    dest_file = tmp_path / "partial.bin"
    initial_content = b"part1_"
    dest_file.write_bytes(initial_content)
    
    # Total file is "part1_part_2" (12 bytes)
    mock_head_res = MagicMock()
    mock_head_res.headers = {
        "content-length": "12",
        "accept-ranges": "bytes"
    }
    
    # Server will return remaining 6 bytes for GET Range request
    mock_get_res = MagicMock()
    mock_get_res.status_code = 206
    mock_get_res.headers = {"content-length": "6"}
    mock_get_res.iter_content.return_value = [b"part_2"]
    mock_get_res.__enter__.return_value = mock_get_res
    
    with patch("requests.head", return_value=mock_head_res):
        with patch("requests.get", return_value=mock_get_res) as mock_get:
            manager.download_file("http://example.com/file.bin", dest_file, "Test Resume")
            
            # Check GET headers contain Range request parameters
            args, kwargs = mock_get.call_args
            assert kwargs["headers"]["Range"] == "bytes=6-"
            
    assert dest_file.read_bytes() == b"part1_part_2"


def test_extract_zip(mock_config, tmp_path):
    """Verifies zip files are extracted successfully to target directory."""
    manager = BinarySetupManager(mock_config)
    
    # Create dummy in-memory zip content
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as z:
        z.writestr("test_file.txt", "zip extraction success contents")
        
    zip_path = tmp_path / "test.zip"
    zip_path.write_bytes(zip_buffer.getvalue())
    
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    
    manager.extract_zip(zip_path, extract_dir)
    
    extracted_file = extract_dir / "test_file.txt"
    assert extracted_file.exists()
    assert extracted_file.read_text() == "zip extraction success contents"


def test_setup_throws_exception_on_missing_exe(mock_config, temp_dirs):
    """Verifies SetupError is raised if the required executable is missing after setup."""
    manager = BinarySetupManager(mock_config)
    manager.ensure_directories()
    
    # Mock download_file and extract_zip to simulate successful runs
    # but do NOT create the mock executable piper.exe
    with patch.object(manager, "download_file") as mock_dl:
        with patch.object(manager, "extract_zip") as mock_unzip:
            with pytest.raises(SetupError) as exc_info:
                manager.setup()
                
            assert "Piper executable was not found" in str(exc_info.value)
            assert mock_dl.call_count >= 1
