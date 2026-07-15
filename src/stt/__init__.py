"""
Speech-to-Text (STT) Subsystem Package for ZovaAI.
Contains interfaces, concrete Whisper.cpp recognizers, and service wrappers.
"""

from src.stt.recognizer import WhisperSpeechRecognizer
from src.stt.service import SpeechRecognitionService

__all__ = [
    "WhisperSpeechRecognizer",
    "SpeechRecognitionService"
]
