"""
Event Subscribers (Plugins) for ZovaAI.
Implements LoggingSubscriber and MetricsSubscriber to handle logging
and performance metrics collection asynchronously in response to events.
"""

import time
from typing import Optional

from src.core.logger import get_logger
from src.metrics.metrics import MetricsCollector
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

logger = get_logger("assistant_subscribers")


class LoggingSubscriber:
    """Subscriber that prints pipeline operations to stdout/logs."""

    def __init__(self, event_bus: EventBus):
        """
        Initializes the logging subscriber.
        
        Args:
            event_bus: The central event bus.
        """
        self.event_bus = event_bus

    def register(self) -> None:
        """Subscribes all logging handlers to event bus."""
        self.event_bus.subscribe(WakeWordDetected, self.on_wake_word_detected)
        self.event_bus.subscribe(RecordingStarted, self.on_recording_started)
        self.event_bus.subscribe(RecordingFinished, self.on_recording_finished)
        self.event_bus.subscribe(STTCompleted, self.on_stt_completed)
        self.event_bus.subscribe(LLMStarted, self.on_llm_started)
        self.event_bus.subscribe(LLMCompleted, self.on_llm_completed)
        self.event_bus.subscribe(TTSStarted, self.on_tts_started)
        self.event_bus.subscribe(TTSCompleted, self.on_tts_completed)
        self.event_bus.subscribe(ErrorOccurred, self.on_error)

    def on_wake_word_detected(self, event: WakeWordDetected) -> None:
        logger.info("Wake word detected")

    def on_recording_started(self, event: RecordingStarted) -> None:
        logger.info("Recording command")

    def on_recording_finished(self, event: RecordingFinished) -> None:
        logger.debug("Recording finished: %s", event.wav_path.name)

    def on_stt_completed(self, event: STTCompleted) -> None:
        logger.info("STT complete: \"%s\"", event.text)

    def on_llm_started(self, event: LLMStarted) -> None:
        logger.info("Sending request to Ollama")

    def on_llm_completed(self, event: LLMCompleted) -> None:
        logger.info("Response received")

    def on_tts_started(self, event: TTSStarted) -> None:
        logger.info("Speaking response")

    def on_tts_completed(self, event: TTSCompleted) -> None:
        logger.info("Ready")

    def on_error(self, event: ErrorOccurred) -> None:
        logger.error("Pipeline error: %s", event.error_message)


class MetricsSubscriber:
    """Subscriber that collects performance and latency metrics centrally."""

    def __init__(self, event_bus: EventBus, metrics: MetricsCollector):
        """
        Initializes the metrics subscriber.
        
        Args:
            event_bus: The central event bus.
            metrics: The metrics collector.
        """
        self.event_bus = event_bus
        self.metrics = metrics
        
        self._llm_start_time: float = 0.0
        self._first_token_time: float = 0.0
        self._tokens_count: int = 0
        self._first_token_received = False

    def register(self) -> None:
        """Subscribes all metrics handlers to event bus."""
        self.event_bus.subscribe(WakeWordDetected, self.on_wake_word_detected)
        self.event_bus.subscribe(RecordingStarted, self.on_recording_started)
        self.event_bus.subscribe(RecordingFinished, self.on_recording_finished)
        self.event_bus.subscribe(STTCompleted, self.on_stt_completed)
        self.event_bus.subscribe(LLMStarted, self.on_llm_started)
        self.event_bus.subscribe(LLMChunkReceived, self.on_llm_chunk_received)
        self.event_bus.subscribe(LLMCompleted, self.on_llm_completed)
        self.event_bus.subscribe(TTSStarted, self.on_tts_started)
        self.event_bus.subscribe(TTSCompleted, self.on_tts_completed)

    def on_wake_word_detected(self, event: WakeWordDetected) -> None:
        self.metrics.clear()
        self.metrics.start_timer("end_to_end_response_time")
        self.metrics.start_timer("wake_word_latency")

    def on_recording_started(self, event: RecordingStarted) -> None:
        self.metrics.stop_timer("wake_word_latency")
        self.metrics.start_timer("recording_duration")

    def on_recording_finished(self, event: RecordingFinished) -> None:
        self.metrics.stop_timer("recording_duration")
        self.metrics.start_timer("stt_latency")

    def on_stt_completed(self, event: STTCompleted) -> None:
        self.metrics.stop_timer("stt_latency")

    def on_llm_started(self, event: LLMStarted) -> None:
        self._llm_start_time = time.perf_counter()
        self._first_token_received = False
        self._tokens_count = 0
        self.metrics.start_timer("llm_total_generation_time")

    def on_llm_chunk_received(self, event: LLMChunkReceived) -> None:
        if not self._first_token_received:
            self._first_token_received = True
            self._first_token_time = time.perf_counter()
            time_to_first = self._first_token_time - self._llm_start_time
            self.metrics.set_value("time_to_first_token", time_to_first)
        
        # Approximate tokens count based on whitespace separation
        chunk_tokens = len(event.chunk.split())
        self._tokens_count += max(1, chunk_tokens)

    def on_llm_completed(self, event: LLMCompleted) -> None:
        generation_time = self.metrics.stop_timer("llm_total_generation_time")
        if generation_time and generation_time > 0:
            # 1 token is approximately 4 characters in normal text datasets
            approx_tokens = len(event.response) / 4.0
            tps = approx_tokens / generation_time
            self.metrics.set_value("tokens_per_second", tps)

    def on_tts_started(self, event: TTSStarted) -> None:
        self.metrics.start_timer("tts_latency")

    def on_tts_completed(self, event: TTSCompleted) -> None:
        self.metrics.stop_timer("tts_latency")
        self.metrics.stop_timer("end_to_end_response_time")
        self.metrics.log_summary()
