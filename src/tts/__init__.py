"""
Text-to-Speech (TTS) Subsystem Package for ZovaAI.
Contains interfaces, concrete Piper synthesizers, and service wrappers.
"""

from src.tts.synthesizer import PiperSpeechSynthesizer
from src.tts.service import SpeechSynthesisService

__all__ = [
    "PiperSpeechSynthesizer",
    "SpeechSynthesisService"
]
