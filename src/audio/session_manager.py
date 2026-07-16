"""
Audio Session Manager for ZovaAI.
Defines Enum-based states and manages thread-safe transitions.
Coordinates microphone, speaker, and orchestrator pipelines.
"""

import threading
from enum import Enum, auto

from src.core.logger import get_logger

logger = get_logger("session_manager")


class SessionState(Enum):
    """Assistant workflow operational states."""
    IDLE = auto()
    LISTENING = auto()  # Idle state waiting for wake word trigger
    RECORDING = auto()  # Voice command recording active
    PROCESSING = auto() # Transcribing speech and querying LLM
    SPEAKING = auto()   # Voice synthesizer vocalization active
    SHUTTING_DOWN = auto() # Application is exiting


class AudioSessionManager:
    """Thread-safe state machine manager for ZovaAI assistant sessions."""

    def __init__(self):
        """Initializes the session state machine to IDLE."""
        self._state = SessionState.IDLE
        self._lock = threading.Lock()

    def get_state(self) -> SessionState:
        """
        Retrieves the current operational state.
        
        Returns:
            SessionState: The active enum state.
        """
        with self._lock:
            return self._state

    def transition_to(self, new_state: SessionState) -> None:
        """
        Attempts a state transition, locking and logging changes.
        
        Args:
            new_state: Destination SessionState enum.
        """
        with self._lock:
            old_state = self._state
            if old_state == new_state:
                return
            
            # Record transition log
            logger.info("Session state transition: %s -> %s", old_state.name, new_state.name)
            self._state = new_state
