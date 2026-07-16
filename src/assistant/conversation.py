"""
Conversation History Manager for ZovaAI.
Maintains a sliding window of recent conversation messages in memory.
"""

from typing import List, Dict


class ConversationManager:
    """Manages the rolling sliding window of chat messages in memory."""

    def __init__(self, max_messages: int = 10):
        """
        Initializes the conversation manager.

        Args:
            max_messages: Maximum user/assistant dialogue turns to preserve.
        """
        self.max_messages = max_messages
        self._history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """
        Adds a message to the dialogue history.

        Args:
            role: Must be 'user', 'assistant', or 'system'.
            content: The text content of the message.

        Raises:
            ValueError: If an unsupported role is provided.
        """
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Unsupported conversation role: {role}")

        self._history.append({"role": role, "content": content})

        # Enforce sliding window capacity limit
        if len(self._history) > self.max_messages:
            self._history = self._history[-self.max_messages:]

    def get_history(self) -> List[Dict[str, str]]:
        """
        Retrieves a copy of the sliding window chat history.

        Returns:
            List[Dict[str, str]]: Copied list of message dictionaries.
        """
        return self._history.copy()

    def clear(self) -> None:
        """Wipes the in-memory chat history."""
        self._history.clear()
