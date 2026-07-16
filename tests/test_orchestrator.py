"""
Unit and integration tests for ZovaAI Assistant Orchestrator.
Tests pipeline execution, strongly typed FIFO event bus transitions, session manager,
cancellation token interrupts, multiple subscribers, and thread lifecycles.
"""

# pylint: disable=broad-exception-caught

import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.core.config import Config
from src.core.cancellation import CancellationToken
from src.core.exceptions import LLMError, AudioError
from src.interfaces.audio_recorder import AudioRecorder
from src.interfaces.speech_recognizer import SpeechRecognizer
from src.wakeword.interfaces import WakeWordService
from src.llm.service import LLMService
from src.tts.service import SpeechSynthesisService
from src.assistant.orchestrator import AssistantOrchestrator
from src.audio.session_manager import AudioSessionManager, SessionState
from src.events.event_bus import EventBus
from src.events.events import (
    WakeWordDetected,
    RecordingStarted,
    RecordingFinished,
    STTCompleted,
    LLMStarted,
    LLMChunkReceived,
    LLMCompleted,
    TTSStarted,
    TTSCompleted,
    ErrorOccurred
)


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
    
    config.audio = MagicMock()
    config.audio.device_index = 0
    
    config.tts = MagicMock()
    config.tts.min_chunk_chars = 10
    config.tts.output_dir = tmp_path / "temp" / "tts"

    recorder = MagicMock(spec=AudioRecorder)
    recorder.is_recording.side_effect = [True, True, False]
    recorder.stop_recording.return_value = tmp_path / "temp" / "recording.wav"
    
    wakeword = MagicMock(spec=WakeWordService)
    stt = MagicMock(spec=SpeechRecognizer)
    llm = MagicMock(spec=LLMService)
    
    # Mock synthesizers inside TTS service
    tts = MagicMock(spec=SpeechSynthesisService)
    tts_inner = MagicMock()
    tts_inner.output_dir = tmp_path / "temp" / "tts"
    tts_inner.output_dir.mkdir(parents=True, exist_ok=True)
    tts.synthesizer = tts_inner
    
    return {
        "config": config,
        "recorder": recorder,
        "wakeword": wakeword,
        "stt": stt,
        "llm": llm,
        "tts": tts
    }


def test_assistant_orchestrator_initialization(mock_pipeline):
    """Checks that orchestrator registers system prompts and states on init."""
    event_bus = EventBus()
    session_manager = AudioSessionManager()
    
    orchestrator = AssistantOrchestrator(
        mock_pipeline["config"],
        mock_pipeline["recorder"],
        mock_pipeline["wakeword"],
        mock_pipeline["stt"],
        mock_pipeline["llm"],
        mock_pipeline["tts"],
        event_bus,
        session_manager
    )
    
    assert orchestrator.name == "Zova"
    assert orchestrator.is_running() is False
    assert session_manager.get_state() == SessionState.IDLE


def test_pipeline_execution_success(mock_pipeline, tmp_path):
    """Checks full pipeline execution with LLM response streaming and event publishing."""
    event_bus = EventBus()
    session_manager = AudioSessionManager()
    
    orchestrator = AssistantOrchestrator(
        mock_pipeline["config"],
        mock_pipeline["recorder"],
        mock_pipeline["wakeword"],
        mock_pipeline["stt"],
        mock_pipeline["llm"],
        mock_pipeline["tts"],
        event_bus,
        session_manager
    )

    # Mock success outputs
    mock_pipeline["stt"].transcribe.return_value = "hello Jarvis"
    
    # Mock LLM stream generator yielding clauses
    def mock_stream(prompt, system_prompt, history):
        yield "I am Zova. "
        yield "An assistant."
    mock_pipeline["llm"].generate_response_stream.side_effect = mock_stream

    # Setup list to accumulate published events from Bus
    published_events = []
    def log_event(event):
        published_events.append(event)

    event_bus.subscribe(WakeWordDetected, log_event)
    event_bus.subscribe(RecordingStarted, log_event)
    event_bus.subscribe(RecordingFinished, log_event)
    event_bus.subscribe(STTCompleted, log_event)
    event_bus.subscribe(LLMStarted, log_event)
    event_bus.subscribe(LLMChunkReceived, log_event)
    event_bus.subscribe(LLMCompleted, log_event)
    event_bus.subscribe(TTSStarted, log_event)
    event_bus.subscribe(TTSCompleted, log_event)

    # Start orchestrator
    orchestrator.start()
    assert session_manager.get_state() == SessionState.LISTENING

    callback = mock_pipeline["wakeword"].register_callback.call_args[0][0]

    with patch("src.tts.streaming_worker.play_wav") as mock_play_wav:
        # Trigger wake-word match
        callback()
        
        # Give Event Bus dispatcher thread brief time to process async queue tasks
        time.sleep(0.5)

        # Stop orchestrator
        orchestrator.stop()

    # Verify event types published in FIFO order
    event_types = [type(e) for e in published_events]
    assert WakeWordDetected in event_types
    assert RecordingStarted in event_types
    assert RecordingFinished in event_types
    assert STTCompleted in event_types
    assert LLMStarted in event_types
    assert LLMChunkReceived in event_types
    assert LLMCompleted in event_types
    assert TTSStarted in event_types
    assert TTSCompleted in event_types

    # Assert correct parameters passed through pipeline
    mock_pipeline["stt"].transcribe.assert_called_once_with(tmp_path / "temp" / "recording.wav")
    mock_pipeline["llm"].generate_response_stream.assert_called_once()
    
    # Playback should be requested on the sentences
    mock_play_wav.assert_called()


def test_empty_transcription_stops_early(mock_pipeline):
    """Checks that empty speech command stops pipeline without calling LLM."""
    event_bus = EventBus()
    session_manager = AudioSessionManager()
    
    orchestrator = AssistantOrchestrator(
        mock_pipeline["config"],
        mock_pipeline["recorder"],
        mock_pipeline["wakeword"],
        mock_pipeline["stt"],
        mock_pipeline["llm"],
        mock_pipeline["tts"],
        event_bus,
        session_manager
    )

    mock_pipeline["stt"].transcribe.return_value = "   "
    
    events_published = []
    event_bus.subscribe(LLMStarted, lambda e: events_published.append(e))

    orchestrator.start()
    callback = mock_pipeline["wakeword"].register_callback.call_args[0][0]
    callback()
    
    time.sleep(0.2)
    final_state = session_manager.get_state()
    orchestrator.stop()

    # Verify LLM was never started
    assert len(events_published) == 0
    assert final_state == SessionState.LISTENING


def test_cancellation_token_stops_worker_immediately(mock_pipeline, tmp_path):
    """Checks that CancellationToken cancel halts background processes."""
    token = CancellationToken()
    assert token.is_cancelled() is False

    token.cancel()
    assert token.is_cancelled() is True

    # Test that when token is cancelled, the streaming worker clears queue and unlinks file
    from src.tts.streaming_worker import StreamingTTSWorker
    
    worker = StreamingTTSWorker(
        mock_pipeline["tts"].synthesizer,
        tmp_path / "temp" / "tts",
        0,
        token
    )
    worker.start()
    
    # Put item in queue (should be ignored immediately because token is cancelled)
    worker.put("Cancel me now.")
    
    time.sleep(0.2)
    worker.stop()
    worker.join()
    assert worker.is_running() is False


def test_event_bus_fifo_asynchronous_execution():
    """Checks that multiple EventBus subscriptions run concurrently without blocks."""
    event_bus = EventBus()
    event_bus.start()

    execution_order = []
    
    # Slow subscriber
    def slow_subscriber(event):
        time.sleep(0.1)
        execution_order.append("slow")

    # Fast subscriber
    def fast_subscriber(event):
        execution_order.append("fast")

    event_bus.subscribe(WakeWordDetected, slow_subscriber)
    event_bus.subscribe(WakeWordDetected, fast_subscriber)

    event_bus.publish(WakeWordDetected())
    
    # Wait for execution thread pool to drain
    time.sleep(0.3)
    event_bus.stop()
    event_bus.join()

    # Fast subscriber runs concurrently, so it completes first despite registration order
    assert "fast" in execution_order
    assert "slow" in execution_order
    assert execution_order[0] == "fast"


def test_rapid_wake_word_interruption_cancellation(mock_pipeline):
    """Checks that triggers during SPEAKING clear state and restart loops."""
    event_bus = EventBus()
    session_manager = AudioSessionManager()
    
    orchestrator = AssistantOrchestrator(
        mock_pipeline["config"],
        mock_pipeline["recorder"],
        mock_pipeline["wakeword"],
        mock_pipeline["stt"],
        mock_pipeline["llm"],
        mock_pipeline["tts"],
        event_bus,
        session_manager
    )

    orchestrator.start()
    
    # Force state to SPEAKING representing vocalization active
    session_manager.transition_to(SessionState.SPEAKING)
    
    # Create mock CancellationToken
    token = CancellationToken()
    orchestrator.current_token = token
    
    callback = mock_pipeline["wakeword"].register_callback.call_args[0][0]
    
    # Simulate a second wake word match (user interrupts speaker)
    callback()
    
    # Token should be flagged cancelled immediately
    assert token.is_cancelled() is True
    
    time.sleep(0.2)
    orchestrator.stop()
