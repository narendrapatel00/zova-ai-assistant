"""
Streaming Text-to-Speech (TTS) Worker for ZovaAI.
Reads text segments from a thread-safe Queue, synthesizes them sequentially
using the local Piper engine, and plays them back. Supports real-time interruption.
"""

import os
import queue
import threading
import uuid
from pathlib import Path
from typing import Optional

import sounddevice as sd  # type: ignore[import-untyped]

from src.core.logger import get_logger
from src.core.cancellation import CancellationToken
from src.core.exceptions import AudioError, TTSError
from src.interfaces.speech_synthesizer import SpeechSynthesizer
from src.audio.playback import play_wav

logger = get_logger("streaming_tts_worker")


class StreamingTTSWorker:
    """Background worker thread processing sentences sequentially for low-latency TTS."""

    def __init__(
        self,
        synthesizer: SpeechSynthesizer,
        output_dir: Path,
        device_index: Optional[int],
        token: CancellationToken
    ):
        """
        Initializes the streaming TTS worker.
        
        Args:
            synthesizer: Underlying SpeechSynthesizer implementation.
            output_dir: Folder to save temporary chunk WAV files.
            device_index: Index of the output audio device.
            token: Shared CancellationToken instance.
        """
        self.synthesizer = synthesizer
        self.output_dir = output_dir
        self.device_index = device_index
        self.token = token
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._current_wav: Optional[Path] = None

    def start(self) -> None:
        """Starts the background processing worker thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="TTSStreamingWorker",
            daemon=True
        )
        self._thread.start()
        logger.info("TTS Streaming Worker thread started.")

    def stop(self) -> None:
        """Stops the worker thread, flushes the queue, and halts active playbacks."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            
        # Put sentinel
        self._queue.put(None)
        
        # Stop sounddevice playbacks instantly
        try:
            sd.stop()
        # pylint: disable=broad-exception-caught
        except Exception:
            pass
            
        self._clear_queue()
        logger.info("TTS Streaming Worker stopped.")

    def join(self, timeout: float = 1.0) -> None:
        """
        Waits for the background worker thread to terminate.
        
        Args:
            timeout: Maximum seconds to block.
        """
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            logger.info("TTS Streaming Worker thread joined.")

    def is_running(self) -> bool:
        """
        Checks if the worker thread is active.
        
        Returns:
            bool: True if running, False otherwise.
        """
        with self._lock:
            return self._running and (self._thread is not None and self._thread.is_alive())

    def put(self, text: str) -> None:
        """
        Queues a text sentence for synthesis and playback.
        
        Args:
            text: Text sentence to read.
        """
        if not self.is_running():
            logger.warning("Streaming worker not running. Ignored speech text: %s", text)
            return
        self._queue.put(text)

    def _clear_queue(self) -> None:
        """Drains all pending sentences from the queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def _worker_loop(self) -> None:
        """Daemon worker loop reading from queue and executing synthesis playbacks."""
        while True:
            try:
                # Check cancellation token
                if self.token.is_cancelled():
                    self._clear_queue()
                
                text = self._queue.get()
                if text is None:
                    # Shutdown sentinel received
                    self._queue.task_done()
                    break
                
                if self.token.is_cancelled():
                    self._queue.task_done()
                    continue
                
                # Generate unique temp file path
                temp_wav = (
                    self.output_dir / f"stream_{uuid.uuid4().hex}.wav"
                )
                self._current_wav = temp_wav
                
                try:
                    logger.debug("Synthesizing chunk: \"%s\"", text)
                    self.synthesizer.synthesize(text, temp_wav)
                    
                    if self.token.is_cancelled():
                        self._cleanup_current_wav()
                        self._queue.task_done()
                        continue
                    
                    logger.debug("Playing chunk: \"%s\"", text)
                    play_wav(
                        temp_wav,
                        device_index=self.device_index
                    )
                    
                except (TTSError, AudioError) as err:
                    logger.error("Error during streaming speech segment: %s", err.message)
                # pylint: disable=broad-exception-caught
                except Exception as e:
                    logger.error("Unexpected error in streaming TTS segment: %s", e)
                finally:
                    self._cleanup_current_wav()
                
                self._queue.task_done()
                
            except Exception as e:
                logger.error("Critical error in Streaming TTS worker loop: %s", e)

    def _cleanup_current_wav(self) -> None:
        """Deletes the active temporary segment WAV file."""
        if self._current_wav and self._current_wav.exists():
            try:
                os.unlink(self._current_wav)
                logger.debug("Deleted temp stream audio: %s", self._current_wav.name)
            # pylint: disable=broad-exception-caught
            except Exception as e:
                logger.warning(
                    "Failed to delete temp stream wav %s: %s",
                    self._current_wav.name, e
                )
            self._current_wav = None
