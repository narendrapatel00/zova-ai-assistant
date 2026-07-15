"""
Configuration Manager for ZovaAI.
Loads config.yaml and handles environment overrides from .env.
Resolves paths relative to the project root dynamically.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import yaml
from dotenv import load_dotenv

from src.core.exceptions import ConfigurationError

# Load environment variables from .env if present
load_dotenv()


@dataclass
class AppConfig:
    name: str
    version: str
    env: str


@dataclass
class LoggingConfig:
    level: str
    log_file: Path
    max_bytes: int
    backup_count: int


@dataclass
class AudioConfig:
    sample_rate: int
    channels: int
    chunk_size: int
    device_index: Optional[int]
    silence_threshold: float
    silence_seconds: float


@dataclass
class WakeWordConfig:
    model_name: str
    threshold: float
    inference_framework: str


@dataclass
class STTConfig:
    model_name: str
    model_dir: Path
    threads: int


@dataclass
class TTSConfig:
    executable_path: Path
    model_path: Path
    config_path: Path
    output_dir: Path


@dataclass
class SetupConfig:
    piper_zip_url: str
    piper_voice_url: str
    piper_voice_config_url: str
    whisper_model_url: str


# pylint: disable=too-many-instance-attributes
class Config:
    """Config manager that loads configurations from config.yaml."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initializes the configuration manager.
        
        Args:
            config_path: Path to config.yaml (optional).
        """
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
        if not config_path:
            config_path = os.getenv("ZOVA_CONFIG_PATH", "config/config.yaml")
        
        self.config_filepath = self.project_root / config_path
        
        # Declare instance attributes for type-safety
        self.app: AppConfig
        self.logging: LoggingConfig
        self.audio: AudioConfig
        self.wake_word: WakeWordConfig
        self.speech_recognition: STTConfig
        self.speech_synthesis: TTSConfig
        self.setup: SetupConfig
        
        self.load()

    def load(self) -> None:
        """
        Loads the yaml file and parses sections into type-safe dataclasses.
        
        Raises:
            ConfigurationError: if config file is missing or invalid.
        """
        if not self.config_filepath.exists():
            raise ConfigurationError(f"Configuration file not found at {self.config_filepath}")
        
        try:
            with open(self.config_filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(f"Failed to parse config YAML: {str(e)}") from e
        
        # 1. Parse App Config
        app_data = data.get("app", {})
        env_override = os.getenv("ZOVA_ENV", app_data.get("env", "development"))
        self.app = AppConfig(
            name=app_data.get("name", "ZovaAI"),
            version=app_data.get("version", "0.1.0"),
            env=env_override
        )
        
        # 2. Parse Logging Config
        log_data = data.get("logging", {})
        log_level = os.getenv("ZOVA_LOG_LEVEL", log_data.get("level", "INFO"))
        raw_log_file = log_data.get("log_file", "logs/assistant.log")
        self.logging = LoggingConfig(
            level=log_level,
            log_file=self.resolve_path(raw_log_file),
            max_bytes=log_data.get("max_bytes", 10485760),
            backup_count=log_data.get("backup_count", 5)
        )
        
        # 3. Parse Audio Config
        audio_data = data.get("audio", {})
        self.audio = AudioConfig(
            sample_rate=audio_data.get("sample_rate", 16000),
            channels=audio_data.get("channels", 1),
            chunk_size=audio_data.get("chunk_size", 1280),
            device_index=audio_data.get("device_index"),
            silence_threshold=audio_data.get("silence_threshold", 0.03),
            silence_seconds=audio_data.get("silence_seconds", 1.5)
        )
        
        # 4. Parse Wake Word Config
        ww_data = data.get("wake_word", {})
        self.wake_word = WakeWordConfig(
            model_name=ww_data.get("model_name", "hey jarvis"),
            threshold=ww_data.get("threshold", 0.5),
            inference_framework=ww_data.get("inference_framework", "onnx")
        )
        
        # 5. Parse STT Config
        stt_data = data.get("speech_recognition", {})
        self.speech_recognition = STTConfig(
            model_name=stt_data.get("model_name", "base.en"),
            model_dir=self.resolve_path(stt_data.get(
                "model_dir", "models/whisper"
            )),
            threads=stt_data.get("threads", 4)
        )
        
        # 6. Parse TTS Config
        tts_data = data.get("speech_synthesis", {})
        self.speech_synthesis = TTSConfig(
            executable_path=self.resolve_path(tts_data.get(
                "executable_path", "bin/piper/piper.exe"
            )),
            model_path=self.resolve_path(tts_data.get(
                "model_path", "models/piper/en_US-lessac-medium.onnx"
            )),
            config_path=self.resolve_path(tts_data.get(
                "config_path", "models/piper/en_US-lessac-medium.onnx.json"
            )),
            output_dir=self.resolve_path(tts_data.get(
                "output_dir", "temp/tts"
            ))
        )
        
        # 7. Parse Setup Config
        setup_data = data.get("setup", {})
        self.setup = SetupConfig(
            piper_zip_url=setup_data.get("piper_zip_url", ""),
            piper_voice_url=setup_data.get("piper_voice_url", ""),
            piper_voice_config_url=setup_data.get("piper_voice_config_url", ""),
            whisper_model_url=setup_data.get("whisper_model_url", "")
        )

    def resolve_path(self, relative_or_absolute_path: str) -> Path:
        """
        Helper method to resolve paths. If the path is relative, it is resolved
        relative to the project root.
        
        Args:
            relative_or_absolute_path: The path string to resolve.
            
        Returns:
            Resolved absolute Path object.
        """
        path = Path(relative_or_absolute_path)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()
