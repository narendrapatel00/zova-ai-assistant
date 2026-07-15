"""
Interface Definitions for ZovaAI.
Follows SOLID principles (Interface Segregation, Dependency Inversion) to define contracts
for core audio, wake word, speech recognition, and speech synthesis services.
"""

from src.interfaces.audio_recorder import AudioRecorder
from src.interfaces.wake_word_detector import WakeWordDetector
from src.interfaces.speech_recognizer import SpeechRecognizer
from src.interfaces.speech_synthesizer import SpeechSynthesizer

__all__ = [
    "AudioRecorder",
    "WakeWordDetector",
    "SpeechRecognizer",
    "SpeechSynthesizer"
]
