"""
LLM Subsystem High-Level Service for ZovaAI.
Wraps the LLMClient interface to coordinate local text generation operations.
"""

from typing import Optional, List, Dict, Generator
from src.llm.ollama_client import LLMClient


class LLMService:
    """Service class encapsulating local LLM operations."""

    def __init__(self, client: LLMClient):
        """
        Initializes the service with an LLMClient.

        Args:
            client: Resolved LLMClient singleton.
        """
        self.client = client

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generates a text completion response.

        Args:
            prompt: User message prompt.
            system_prompt: Optional system prompt context.
            history: Optional sliding window chat history list.

        Returns:
            str: Assistant text response.
        """
        return self.client.generate(prompt, system_prompt, history)

    def generate_response_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, None]:
        """
        Streams chat response tokens.

        Args:
            prompt: User message prompt.
            system_prompt: Optional system prompt context.
            history: Optional sliding chat history.

        Yields:
            str: Streamed tokens.
        """
        yield from self.client.generate_stream(prompt, system_prompt, history)
