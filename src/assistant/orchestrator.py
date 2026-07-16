"""
Assistant Orchestrator for ZovaAI.
Refactored to support strongly typed Event Bus publishing, shared CancellationToken,
Enum-based state machine transitions, and low-latency chunk-by-sentence TTS vocalizations.
"""

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments
# pylint: disable=broad-exception-caught

import time
from typing import List, Optional

from src.core.config import Config
from src.core.logger import get_logger
from src.core.exceptions import AudioError, WakeWordError, STTError, TTSError, LLMError
from src.core.cancellation import CancellationToken
from src.interfaces.audio_recorder import AudioRecorder
from src.interfaces.speech_recognizer import SpeechRecognizer
from src.wakeword.interfaces import WakeWordService
from src.llm.service import LLMService
from src.tts.service import SpeechSynthesisService
from src.tts.streaming_worker import StreamingTTSWorker
from src.audio.session_manager import AudioSessionManager, SessionState
from src.assistant.conversation import ConversationManager
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

logger = get_logger("assistant_orchestrator")


class AssistantOrchestrator:
    """Central manager driving the low-latency offline voice pipeline loops."""

    def __init__(
        self,
        config: Config,
        audio_recorder: AudioRecorder,
        wakeword_service: WakeWordService,
        speech_recognizer: SpeechRecognizer,
        llm_service: LLMService,
        speech_synthesizer: SpeechSynthesisService,
        event_bus: EventBus,
        session_manager: AudioSessionManager
    ):
        """
        Initializes the assistant orchestrator.
        
        Args:
            config: Loaded application config manager.
            audio_recorder: AudioRecorder stream service.
            wakeword_service: WakeWord detector background service.
            speech_recognizer: STT transcriber service.
            llm_service: LLM inference service.
            speech_synthesizer: TTS synthesis service.
            event_bus: Decoupled pub/sub event bus.
            session_manager: Audio session state coordinator.
        """
        self.config = config
        self.audio_recorder = audio_recorder
        self.wakeword_service = wakeword_service
        self.speech_recognizer = speech_recognizer
        self.llm_service = llm_service
        self.speech_synthesizer = speech_synthesizer
        self.event_bus = event_bus
        self.session_manager = session_manager

        self.name = config.assistant.name
        self.wake_response = config.assistant.wake_response
        self.system_prompt_path = config.assistant.system_prompt

        self.conversation_manager = ConversationManager(max_messages=10)
        self.current_token: Optional[CancellationToken] = None
        self.tts_worker: Optional[StreamingTTSWorker] = None
        self._running = False
        self._system_prompt = ""

    def _load_system_prompt(self) -> str:
        """Loads system prompt from disk, generating a default if missing."""
        try:
            resolved_path = self.config.resolve_path(str(self.system_prompt_path))
            if not resolved_path.exists():
                resolved_path.parent.mkdir(parents=True, exist_ok=True)
                default_prompt = (
                    f"You are {self.name}, a helpful local offline personal AI assistant. "
                    "Keep your responses short, friendly, and conversational."
                )
                resolved_path.write_text(default_prompt, encoding="utf-8")
                logger.info("Generated default system prompt at %s", resolved_path)
                return default_prompt
            
            prompt = resolved_path.read_text(encoding="utf-8").strip()
            logger.info("System prompt loaded from %s", resolved_path)
            return prompt
        except Exception as e:
            logger.warning("Failed to load system prompt: %s. Using default.", e)
            return f"You are {self.name}, a helpful personal assistant."

    def start(self) -> None:
        """Starts the event bus dispatcher and hooks wake word loops."""
        if self._running:
            logger.warning("AssistantOrchestrator is already running.")
            return

        self._running = True
        self._system_prompt = self._load_system_prompt()
        
        # Start event bus thread first
        self.event_bus.start()
        
        # Transition state to listening
        self.session_manager.transition_to(SessionState.LISTENING)
        
        # Register wake word callback and start background thread
        self.wakeword_service.register_callback(self._on_wake_word_trigger)
        self.wakeword_service.start()
        
        logger.info("Assistant Orchestrator active and listening.")

    def stop(self) -> None:
        """Gracefully stops all dispatcher threads, workers, and queues."""
        if not self._running:
            return
        
        self._running = False
        self.session_manager.transition_to(SessionState.SHUTTING_DOWN)
        
        # Cancel any active execution cycles
        if self.current_token:
            self.current_token.cancel()
            
        # Stop background threads
        self.wakeword_service.stop()
        if self.tts_worker:
            self.tts_worker.stop()
            self.tts_worker.join(timeout=1.0)
            
        self.event_bus.stop()
        self.event_bus.join(timeout=1.0)
        logger.info("Assistant Orchestrator stopped.")

    def is_running(self) -> bool:
        """Checks if the assistant orchestrator is active."""
        return self._running

    def _on_wake_word_trigger(self) -> None:
        """Callback triggered when the wake word is detected."""
        if not self._running:
            return

        # Handle cancellation if wake word matched during active vocalization
        active_state = self.session_manager.get_state()
        if active_state == SessionState.SPEAKING or active_state == SessionState.PROCESSING:
            logger.info("User interrupted speaking assistant. Cancelling current cycle.")
            if self.current_token:
                self.current_token.cancel()
            if self.tts_worker:
                self.tts_worker.stop()
            
            # Short sleep to clear buffers and sockets
            time.sleep(0.2)
            
        # Transition to recording state
        self.session_manager.transition_to(SessionState.RECORDING)
        self.event_bus.publish(WakeWordDetected())

        # Spawn new clean cancellation token and TTS worker for this run
        self.current_token = CancellationToken()
        self.tts_worker = StreamingTTSWorker(
            self.speech_synthesizer.synthesizer,
            self.config.tts.output_dir,
            self.config.audio.device_index,
            self.current_token
        )
        self.tts_worker.start()

        # Run pipeline in a safe execution block
        try:
            self._execute_pipeline(self.current_token, self.tts_worker)
        except Exception as e:
            logger.error("Error running orchestrator pipeline: %s", e)
            self.event_bus.publish(ErrorOccurred(error_message=str(e)))
            
            # Transition state back to listening
            if self._running:
                self.session_manager.transition_to(SessionState.LISTENING)

    def _execute_pipeline(self, token: CancellationToken, tts_worker: StreamingTTSWorker) -> None:
        """Runs the sequence steps checking the cancellation token between blocks."""
        # 1. Record voice command
        self.event_bus.publish(RecordingStarted())
        
        self.audio_recorder.start_recording()
        while self.audio_recorder.is_recording():
            if token.is_cancelled():
                break
            self.audio_recorder.get_audio_chunk()
            
        wav_path = self.audio_recorder.stop_recording()
        
        if token.is_cancelled():
            logger.info("Command cycle cancelled during voice recording.")
            return

        self.event_bus.publish(RecordingFinished(wav_path=wav_path))

        # 2. Transcribe WAV file to text
        self.session_manager.transition_to(SessionState.PROCESSING)
        time.sleep(0.1)  # Release file locks
        transcribed_text = self.speech_recognizer.transcribe(wav_path)
        
        if token.is_cancelled():
            logger.info("Command cycle cancelled during STT transcription.")
            return

        if not transcribed_text.strip():
            logger.info("Empty command transcription. Resetting state.")
            self.session_manager.transition_to(SessionState.LISTENING)
            self.event_bus.publish(TTSCompleted())  # Signal end
            return

        self.event_bus.publish(STTCompleted(text=transcribed_text))

        # 3. Stream Ollama Response and accumulate sentences
        self.event_bus.publish(LLMStarted(prompt=transcribed_text))
        
        history = self.conversation_manager.get_history()
        stream = self.llm_service.generate_response_stream(
            transcribed_text,
            self._system_prompt,
            history
        )
        
        buffer = ""
        full_response = ""
        min_chars = self.config.tts.min_chunk_chars
        sentence_delimiters = (".", "?", "!", "\n")

        for chunk in stream:
            if token.is_cancelled():
                break
            
            self.event_bus.publish(LLMChunkReceived(chunk=chunk))
            buffer += chunk
            full_response += chunk
            
            # Extract sentence boundaries
            min_idx = -1
            for delim in sentence_delimiters:
                idx = buffer.find(delim)
                if idx != -1:
                    if min_idx == -1 or idx < min_idx:
                        min_idx = idx
            
            if min_idx != -1:
                sentence = buffer[:min_idx + 1]
                if len(sentence.strip()) >= min_chars:
                    self.session_manager.transition_to(SessionState.SPEAKING)
                    tts_worker.put(sentence.strip())
                    buffer = buffer[min_idx + 1:]

        # Abort if cancelled
        if token.is_cancelled():
            logger.info("Command cycle cancelled during LLM stream.")
            return

        # Dispatch any remaining text buffer
        if buffer.strip():
            self.session_manager.transition_to(SessionState.SPEAKING)
            tts_worker.put(buffer.strip())

        self.event_bus.publish(LLMCompleted(response=full_response))
        
        # Save to memory history
        self.conversation_manager.add_message("user", transcribed_text)
        self.conversation_manager.add_message("assistant", full_response)

        # 4. Wait for all speech segments to play back
        self.event_bus.publish(TTSStarted(text=full_response))
        
        # Blocks orchestrator thread until TTS Queue is completely drained
        tts_worker._queue.join()
        
        if token.is_cancelled():
            logger.info("Command cycle cancelled during TTS vocalization.")
            return
            
        self.event_bus.publish(TTSCompleted())
        
        # Re-enter waiting for wake word state
        if self._running:
            self.session_manager.transition_to(SessionState.LISTENING)
