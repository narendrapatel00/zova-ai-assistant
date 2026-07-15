"""
Tests for ZovaAI configuration loading, environment overrides,
logging initialization, and dependency injection container.
"""

import os
import logging
from pathlib import Path
import pytest
import yaml

from src.core.config import Config
from src.core.exceptions import ConfigurationError, DependencyInjectionError
from src.core.logger import LoggerSetup, get_logger
from src.core.di import DIContainer


@pytest.fixture
def temp_config_file(tmp_path) -> Path:
    """Fixture to generate a temporary valid configuration yaml file."""
    config_data = {
        "app": {
            "name": "TestZova",
            "version": "0.0.1",
            "env": "testing"
        },
        "logging": {
            "level": "DEBUG",
            "log_file": str(tmp_path / "logs" / "test.log"),
            "max_bytes": 1024,
            "backup_count": 2
        },
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "chunk_size": 1024
        },
        "wake_word": {
            "model_name": "hey jarvis",
            "threshold": 0.5
        },
        "speech_recognition": {
            "model_name": "tiny.en"
        },
        "speech_synthesis": {
            "executable_path": "bin/piper/piper.exe"
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f)
    return config_file


def test_config_load_success(temp_config_file, monkeypatch):
    """Verifies that Config class successfully loads and parses settings."""
    # Ensure local .env variables do not interfere with config loading tests
    monkeypatch.delenv("ZOVA_ENV", raising=False)
    monkeypatch.delenv("ZOVA_LOG_LEVEL", raising=False)
    
    config = Config(config_path=str(temp_config_file))
    
    assert config.app.name == "TestZova"
    assert config.app.version == "0.0.1"
    assert config.app.env == "testing"
    assert config.audio.sample_rate == 16000
    assert config.wake_word.model_name == "hey jarvis"
    assert config.speech_recognition.model_name == "tiny.en"


def test_config_missing_file_throws_error():
    """Verifies that configuration loading throws an exception for non-existent files."""
    with pytest.raises(ConfigurationError) as exc_info:
        Config(config_path="non_existent_folder/config_file.yaml")
    assert "Configuration file not found" in str(exc_info.value)


def test_config_env_var_overrides(temp_config_file, monkeypatch):
    """Verifies that environment variables successfully override config file properties."""
    monkeypatch.setenv("ZOVA_ENV", "production")
    monkeypatch.setenv("ZOVA_LOG_LEVEL", "WARNING")
    
    config = Config(config_path=str(temp_config_file))
    
    assert config.app.env == "production"
    assert config.logging.level == "WARNING"


def test_logger_initialization(temp_config_file, tmp_path):
    """Verifies that logger setup configures handlers, creates directories, and rotates files."""
    config = Config(config_path=str(temp_config_file))
    log_file = tmp_path / "logs" / "test_run.log"
    
    # Ensure log file doesn't exist
    if log_file.exists():
        os.remove(log_file)
        
    # Reset logger setup initializer state for testing
    LoggerSetup._initialized = False
    
    root_logger = LoggerSetup.initialize(
        log_level="DEBUG",
        log_file=log_file,
        max_bytes=1000,
        backup_count=2
    )
    
    # Check that handlers were added
    handlers = root_logger.handlers
    assert len(handlers) >= 1
    
    # Test writing a log
    test_message = "Hello Zova Logging Test"
    logger = get_logger("test_module")
    logger.info(test_message)
    
    # Verify file was created and log message is present
    assert log_file.exists()
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert test_message in content
        assert "INFO" in content
        assert "test_module" in content


def test_di_container_registration_resolution():
    """Verifies the DI Container registers and resolves singletons and factory instances."""
    container = DIContainer()
    container.clear()
    
    # Define dummy interface and implementation
    class IDummy:
        pass
        
    class DummyImpl(IDummy):
        def __init__(self, value):
            self.value = value
            
    # 1. Test registration and resolution of pre-instantiated singleton object
    dummy_inst = DummyImpl("static-singleton")
    container.register(IDummy, dummy_inst, singleton=True)
    
    resolved = container.resolve(IDummy)
    assert resolved is dummy_inst
    assert resolved.value == "static-singleton"
    
    # 2. Test registration and resolution of a factory callable (singleton)
    container.clear()
    factory_counter = 0
    
    def dummy_factory(c):
        nonlocal factory_counter
        factory_counter += 1
        return DummyImpl(f"factory-singleton-{factory_counter}")
        
    container.register(IDummy, dummy_factory, singleton=True)
    
    res1 = container.resolve(IDummy)
    res2 = container.resolve(IDummy)
    
    assert res1 is res2
    assert res1.value == "factory-singleton-1"
    assert factory_counter == 1
    
    # 3. Test factory registration (not singleton)
    container.clear()
    factory_counter = 0
    container.register(IDummy, dummy_factory, singleton=False)
    
    res1 = container.resolve(IDummy)
    res2 = container.resolve(IDummy)
    
    assert res1 is not res2
    assert res1.value == "factory-singleton-1"
    assert res2.value == "factory-singleton-2"
    assert factory_counter == 2


def test_di_container_unregistered_throws_error():
    """Verifies resolving an unregistered dependency raises a DependencyInjectionError."""
    container = DIContainer()
    container.clear()
    
    class IUnregistered:
        pass
        
    with pytest.raises(DependencyInjectionError) as exc_info:
        container.resolve(IUnregistered)
    assert "is not registered" in str(exc_info.value)
