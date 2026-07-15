"""
openWakeWord Engine Detector Implementation for ZovaAI.
Implements the WakeWordDetector interface. Loads ONNX models and performs local
inference on 16kHz mono audio streams.
"""

from typing import Optional
import numpy as np
from openwakeword.model import Model  # type: ignore[import-untyped]

from src.core.config import Config
from src.core.logger import get_logger
from src.core.exceptions import WakeWordError
from src.interfaces.wake_word_detector import WakeWordDetector

logger = get_logger("wakeword_engine")


class OpenWakeWordDetector(WakeWordDetector):
    """Concrete implementation of WakeWordDetector interface wrapping openWakeWord."""

    def __init__(self, config: Config):
        """
        Initializes the openWakeWord inference model.
        
        Args:
            config: Loaded application configuration manager.
            
        Raises:
            WakeWordError: If model file cannot be found or ONNX session fails.
        """
        self.config = config
        self.model_path = config.wakeword.model_path
        self.threshold = config.wakeword.threshold
        self._oww_model: Optional[Model] = None
        self.model_name = "hey_jarvis"
        
        self._load_model()

    def _load_model(self) -> None:
        """Loads default 'hey_jarvis' model or custom model path into memory."""
        try:
            if not self.model_path:
                logger.info("Loading default openWakeWord model: 'hey_jarvis'")
                self._oww_model = Model(wakeword_models=["hey_jarvis"])
                self.model_name = "hey_jarvis"
            else:
                resolved_path = self.config.resolve_path(self.model_path)
                if not resolved_path.exists():
                    raise WakeWordError(f"Custom wake-word model path not found: {resolved_path}")
                
                logger.info("Loading custom wake-word model path: %s", resolved_path)
                self._oww_model = Model(wakeword_models=[str(resolved_path)])
                self.model_name = resolved_path.stem
                
            logger.info("openWakeWord model loaded successfully.")
        except Exception as e:
            if not isinstance(e, WakeWordError):
                raise WakeWordError(f"Failed to load openWakeWord model: {str(e)}") from e
            raise e

    def detect(self, chunk: np.ndarray) -> bool:
        """
        Runs local ONNX inference on the audio chunk.
        
        Args:
            chunk: Numpy array of 1280 samples (16kHz 16-bit PCM).
            
        Returns:
            bool: True if prediction exceeds threshold and cooldown is inactive.
        """
        if not self._oww_model:
            raise WakeWordError("openWakeWord model is not loaded.")
            
        try:
            # openWakeWord predict returns a dict of model names mapping to score floats
            prediction = self._oww_model.predict(chunk)
            
            for name, score in prediction.items():
                if score >= self.threshold:
                    logger.info(
                        "Wake word detected: %s (confidence=%.2f, threshold=%.2f)",
                        name, score, self.threshold
                    )
                    return True
            return False
            
        except Exception as e:
            raise WakeWordError(f"ONNX inference execution failed: {str(e)}") from e

    def get_wake_word_name(self) -> str:
        """Gets the loaded model's filename stem."""
        return self.model_name

    def close(self) -> None:
        """Releases the underlying ONNX runtime inference session."""
        logger.info("Releasing openWakeWord engine runtime session...")
        self._oww_model = None
