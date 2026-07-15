"""
Wake Word Subsystem Package for ZovaAI.
Contains interfaces, concrete openWakeWord detectors, and listener services.
"""

from src.wakeword.interfaces import WakeWordService
from src.wakeword.engine import OpenWakeWordDetector
from src.wakeword.service import WakeWordListeningService

__all__ = [
    "WakeWordService",
    "OpenWakeWordDetector",
    "WakeWordListeningService"
]
