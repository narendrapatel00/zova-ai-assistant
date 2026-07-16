"""
Cancellation Token Implementation for ZovaAI.
Allows thread-safe signaling of cancellation requests across asynchronous tasks.
"""

import threading


class CancellationToken:
    """Thread-safe cancellation coordinator shared across services."""

    def __init__(self):
        """Initializes the cancellation token to active/uncancelled state."""
        self._is_cancelled = False
        self._lock = threading.Lock()

    def cancel(self) -> None:
        """Flags the token as cancelled."""
        with self._lock:
            self._is_cancelled = True

    def is_cancelled(self) -> bool:
        """
        Checks if a cancellation signal has been published.

        Returns:
            bool: True if cancelled, False otherwise.
        """
        with self._lock:
            return self._is_cancelled

    def reset(self) -> None:
        """Resets the cancellation flag to false."""
        with self._lock:
            self._is_cancelled = False
