"""
Unit tests for ZovaAI Assistant Orchestrator.
Tests pipeline execution, event observer callbacks, empty transcription exits,
Ollama connections, and pipeline error recovery using mocks.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.core.config import Config
from src.core.exceptions import LLMError, AudioError
from src.interfaces.audio_recorder import AudioRecorder
from src.interfaces.speech_recognizer import SpeechRecognizer
from src.wakeword.interfaces import WakeWordService
from src.llm.service import LLMService
from src.tts.service import SpeechSynthesisService
from src.assistant.orchestrator import AssistantOrchestrator
from src.assistant.events import AssistantEventObserver


@pytest.fixture
def mock_pipeline(tmp_path):
    """Fixture to build mock pipeline dependencies."""
    config = MagicMock(spec=Config)
    config.project_root = tmp_path
    
    config.assistant = MagicMock()
    config.assistant.name = "Zova"
    config.assistant.wake_response = "Yes?"
    config.assistant.system_prompt = tmp_path / "prompts" / "system.txt"
    config.resolve_path.side_effect = lambda p: Path(p)

    recorder = MagicMock(spec=AudioRecorder)
    # Simulate a stream that captures 2 chunks then finishes (VAD stops)
    recorder.is_recording.side_effect = [True, True, False]
    recorder.stop_recording.return_value = tmp_path / "temp" / "recording.wav"
    
    wakeword = MagicMock(spec=WakeWordService)
    stt = MagicMock(spec=SpeechRecognizer)
    llm = MagicMock(spec=LLMService)
    tts = MagicMock(spec=SpeechSynthesisService)
    
    return {
        "config": config,
        "recorder": recorder,
        "wakeword": wakeword,
        "stt": stt,
        "llm": llm,
        "tts": tts
    }


def test_assistant_orchestrator_initialization(mock_pipeline):
    """Checks that orchestrator registers system prompts and observer bindings."""
    orchestrator = AssistantOrchestrator(
        mock_pipeline["config"],
        mock_pipeline["recorder"],
        mock_pipeline["wakeword"],
        mock_pipeline["stt"],
        mock_pipeline["llm"],
        mock_pipeline["tts"]
    )
    
    assert orchestrator.name == "Zova"
    assert orchestrator.is_running() is False


def test_pipeline_execution_success(mock_pipeline, tmp_path):
    """Checks the full wake word -> record -> STT -> LLM -> TTS sequence."""
    orchestrator = AssistantOrchestrator(
        mock_pipeline["config"],
        mock_pipeline["recorder"],
        mock_pipeline["wakeword"],
        mock_pipeline["stt"],
        mock_pipeline["llm"],
        mock_pipeline["tts"]
    )

    # 1. Setup transcription and LLM response values
    mock_pipeline["stt"].transcribe.return_value = "hello Jarvis who are you"
    mock_pipeline["llm"].generate_response.return_value = "I am Zova, your assistant."

    # 2. Register mock observer to track lifecycle events
    observer = MagicMock(spec=AssistantEventObserver)
    orchestrator.register_observer(observer)

    # 3. Start orchestrator (stores callback registration ref)
    orchestrator.start()
    
    # Extract the wake word callback registered
    mock_pipeline["wakeword"].register_callback.assert_called_once()
    callback = mock_pipeline["wakeword"].register_callback.call_args[0][0]

    # Verify initial wait state notified
    observer.on_waiting_for_wake_word.assert_called_once()
    observer.on_waiting_for_wake_word.reset_mock()

    # 4. Trigger the wake word callback (runs processing cycle)
    callback()

    # 5. Assert pipeline step invocations
    mock_pipeline["recorder"].start_recording.assert_called_once()
    mock_pipeline["recorder"].stop_recording.assert_called_once()
    
    mock_pipeline["stt"].transcribe.assert_called_once_with(tmp_path / "temp" / "recording.wav")
    
    # Check LLM call includes system prompt and chat history
    mock_pipeline["llm"].generate_response.assert_called_once()
    args = mock_pipeline["llm"].generate_response.call_args[0]
    assert args[0] == "hello Jarvis who are you"
    assert "You are Zova" in args[1] # System prompt text
    assert len(args[2]) == 0 # Chat history (initially empty)

    # Check TTS playback matches LLM response
    mock_pipeline["tts"].speak.assert_called_once_with("I am Zova, your assistant.")

    # 6. Verify observer event transitions
    observer.on_wake_word_detected.assert_called_once()
    observer.on_recording_started.assert_called_once()
    observer.on_recording_finished.assert_called_once_with(tmp_path / "temp" / "recording.wav")
    observer.on_transcription_finished.assert_called_once_with("hello Jarvis who are you")
    observer.on_llm_start.assert_called_once_with("hello Jarvis who are you")
    observer.on_llm_response.assert_called_once_with("I am Zova, your assistant.")
    observer.on_tts_start.assert_called_once_with("I am Zova, your assistant.")
    observer.on_tts_finished.assert_called_once()
    observer.on_waiting_for_wake_word.assert_called_once()
    
    # Assert conversation history was updated
    history = orchestrator.conversation_manager.get_history()
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "hello Jarvis who are you"}
    assert history[1] == {"role": "assistant", "content": "I am Zova, your assistant."}


def test_empty_transcription_exits_early(mock_pipeline):
    """Checks that empty speech input skips the LLM and TTS segments."""
    orchestrator = AssistantOrchestrator(
        mock_pipeline["config"],
        mock_pipeline["recorder"],
        mock_pipeline["wakeword"],
        mock_pipeline["stt"],
        mock_pipeline["llm"],
        mock_pipeline["tts"]
    )

    # Mock empty STT transcription
    mock_pipeline["stt"].transcribe.return_value = "   "

    observer = MagicMock(spec=AssistantEventObserver)
    orchestrator.register_observer(observer)

    orchestrator.start()
    callback = mock_pipeline["wakeword"].register_callback.call_args[0][0]
    callback()

    # Verify STT was called but LLM and TTS were bypassed
    mock_pipeline["stt"].transcribe.assert_called_once()
    mock_pipeline["llm"].generate_response.assert_not_called()
    mock_pipeline["tts"].speak.assert_not_called()
    
    # Verify events
    observer.on_transcription_finished.assert_not_called()
    observer.on_llm_start.assert_not_called()


def test_ollama_unavailable_recovery(mock_pipeline):
    """Checks that LLM connection failures are spoken to users and observers receive on_error."""
    orchestrator = AssistantOrchestrator(
        mock_pipeline["config"],
        mock_pipeline["recorder"],
        mock_pipeline["wakeword"],
        mock_pipeline["stt"],
        mock_pipeline["llm"],
        mock_pipeline["tts"]
    )

    mock_pipeline["stt"].transcribe.return_value = "How does this work"
    # Raise LLM server offline error
    mock_pipeline["llm"].generate_response.side_effect = LLMError("Local Ollama connection refused.")

    observer = MagicMock(spec=AssistantEventObserver)
    orchestrator.register_observer(observer)

    orchestrator.start()
    callback = mock_pipeline["wakeword"].register_callback.call_args[0][0]
    callback()

    # Verify observer was notified of the error
    observer.on_error.assert_called_once_with("Local Ollama connection refused.")
    
    # Verify that assistant vocalized the connection failure to the speaker
    mock_pipeline["tts"].speak.assert_called_once()
    assert "verify that Ollama is running" in mock_pipeline["tts"].speak.call_args[0][0]


def test_audio_hardware_crash_logs_error(mock_pipeline):
    """Checks that audio hardware errors are handled gracefully and logged."""
    orchestrator = AssistantOrchestrator(
        mock_pipeline["config"],
        mock_pipeline["recorder"],
        mock_pipeline["wakeword"],
        mock_pipeline["stt"],
        mock_pipeline["llm"],
        mock_pipeline["tts"]
    )

    # Raise hardware error on start recording
    mock_pipeline["recorder"].start_recording.side_effect = AudioError("Failed to open audio stream.")

    observer = MagicMock(spec=AssistantEventObserver)
    orchestrator.register_observer(observer)

    orchestrator.start()
    callback = mock_pipeline["wakeword"].register_callback.call_args[0][0]
    callback()

    # Verify error callback and graceful state reset
    observer.on_error.assert_called_once_with("Failed to open audio stream.")
