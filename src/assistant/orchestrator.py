"""
Central Assistant Orchestrator for ZovaAI.
Coordinates state transitions and data flows between components:
AudioRecorder -> WakeWord -> STT -> Conversation -> LLM -> TTS.
"""

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments
# pylint: disable=broad-exception-caught

import time
from typing import List
from src.core.config import Config
from src.core.logger import get_logger
from src.core.exceptions import AudioError, WakeWordError, STTError, TTSError, LLMError
from src.interfaces.audio_recorder import AudioRecorder
from src.interfaces.speech_recognizer import SpeechRecognizer
from src.wakeword.interfaces import WakeWordService
from src.llm.service import LLMService
from src.tts.service import SpeechSynthesisService
from src.assistant.conversation import ConversationManager
from src.assistant.events import AssistantEventObserver

logger = get_logger("assistant_orchestrator")


class AssistantOrchestrator:
    """Core orchestrator that binds speech, wakeword, LLM, and synthesis engines together."""

    def __init__(
        self,
        config: Config,
        audio_recorder: AudioRecorder,
        wakeword_service: WakeWordService,
        speech_recognizer: SpeechRecognizer,
        llm_service: LLMService,
        speech_synthesizer: SpeechSynthesisService
    ):
        """
        Initializes the assistant orchestrator.
        
        Args:
            config: Central configuration manager.
            audio_recorder: Resolved AudioRecorder singleton.
            wakeword_service: Resolved WakeWordListeningService singleton.
            speech_recognizer: Resolved SpeechRecognizer singleton.
            llm_service: Resolved LLMService singleton.
            speech_synthesizer: Resolved SpeechSynthesisService singleton.
        """
        self.config = config
        self.audio_recorder = audio_recorder
        self.wakeword_service = wakeword_service
        self.speech_recognizer = speech_recognizer
        self.llm_service = llm_service
        self.speech_synthesizer = speech_synthesizer

        self.name = config.assistant.name
        self.wake_response = config.assistant.wake_response
        self.system_prompt_path = config.assistant.system_prompt

        self.conversation_manager = ConversationManager(max_messages=10)
        self._observers: List[AssistantEventObserver] = []
        self._running = False
        self._system_prompt = ""

    def register_observer(self, observer: AssistantEventObserver) -> None:
        """Registers a lifecycle observer to listen to pipeline transitions."""
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: AssistantEventObserver) -> None:
        """Removes a registered lifecycle observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify(self, event_name: str, *args, **kwargs) -> None:
        """Helper to fire observer hooks safely without raising execution blocks."""
        for obs in self._observers:
            try:
                callback = getattr(obs, event_name, None)
                if callback:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "Error in observer %s callback %s: %s",
                    type(obs).__name__, event_name, e
                )

    def _load_system_prompt(self) -> str:
        """Loads system prompt instructions from system_prompt_path. Creates default if missing."""
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
            logger.warning("Failed to load system prompt: %s. Using basic default.", e)
            return f"You are {self.name}, a helpful personal assistant."

    def start(self) -> None:
        """Starts the assistant orchestration lifecycle loop."""
        if self._running:
            logger.warning("AssistantOrchestrator is already running.")
            return

        self._running = True
        self._system_prompt = self._load_system_prompt()
        
        # Register the processing callback to wake-word triggers
        self.wakeword_service.register_callback(self._on_wake_word_trigger)
        self.wakeword_service.start()
        
        logger.info("Assistant Orchestrator active.")
        self._notify("on_waiting_for_wake_word")

    def stop(self) -> None:
        """Stops the orchestrator and background services."""
        if not self._running:
            return
        
        self._running = False
        self.wakeword_service.stop()
        logger.info("Assistant Orchestrator stopped.")

    def is_running(self) -> bool:
        """Checks if orchestrator loop is active."""
        return self._running

    def _on_wake_word_trigger(self) -> None:
        """Internal callback executed on background wake word thread detections."""
        if not self._running:
            return

        logger.info("Wake word matched! Processing command loop...")
        self._notify("on_wake_word_detected")

        try:
            # 1. Capture User Voice Command
            self._notify("on_recording_started")
            logger.info("Recording command...")
            
            self.audio_recorder.start_recording()
            
            # Read chunks. blocks until VAD silence detection triggers stop
            while self.audio_recorder.is_recording():
                self.audio_recorder.get_audio_chunk()

            wav_path = self.audio_recorder.stop_recording()
            logger.info("Recording finished: %s", wav_path.name)
            self._notify("on_recording_finished", wav_path)

            # 2. Transcribe voice file to text
            logger.info("Transcribing audio...")
            # Brief delay to ensure file lock is released
            time.sleep(0.1)
            transcribed_text = self.speech_recognizer.transcribe(wav_path)
            
            if not transcribed_text.strip():
                logger.info("Empty transcription. Skipping LLM generation.")
                self._notify("on_waiting_for_wake_word")
                return

            self._notify("on_transcription_finished", transcribed_text)

            # 3. Generate response using Ollama
            logger.info("Sending request to LLM...")
            self._notify("on_llm_start", transcribed_text)
            
            history = self.conversation_manager.get_history()
            response_text = self.llm_service.generate_response(
                transcribed_text,
                self._system_prompt,
                history
            )
            
            logger.info("LLM Response: \"%s\"", response_text)
            self._notify("on_llm_response", response_text)
            
            # Update dialogue history memory
            self.conversation_manager.add_message("user", transcribed_text)
            self.conversation_manager.add_message("assistant", response_text)

            # 4. Synthesize voice response
            logger.info("Speaking response...")
            self._notify("on_tts_start", response_text)
            
            self.speech_synthesizer.speak(response_text)
            
            self._notify("on_tts_finished")

        except (AudioError, WakeWordError, STTError, TTSError, LLMError) as err:
            logger.error("Assistant Pipeline Error: %s", err.message)
            self._notify("on_error", err.message)
            
            # If Ollama server is offline, vocalize error directly to speakers as feedback
            if isinstance(err, LLMError):
                try:
                    self.speech_synthesizer.speak(
                        "Sorry, I cannot connect to the local brain server. "
                        "Please verify that Ollama is running."
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.error("Unhandled pipeline exception: %s", e)
            self._notify("on_error", str(e))
        finally:
            # Re-enter waiting for wake word state
            if self._running:
                logger.info("Ready. Waiting for wake word...")
                self._notify("on_waiting_for_wake_word")
