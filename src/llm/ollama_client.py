"""
Ollama Local LLM Client Implementation for ZovaAI.
Defines the LLMClient interface and provides a concrete implementation that
queries local Ollama server chat completion API endpoints.
"""

import json
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Generator
import requests

from src.core.config import Config
from src.core.logger import get_logger
from src.core.exceptions import LLMError

logger = get_logger("llm_client")


class LLMClient(ABC):
    """Abstract interface defining the contract for local LLM text generators."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generates a text completion response.

        Args:
            prompt: User message prompt.
            system_prompt: Optional system prompt context instructions.
            history: Optional sliding window chat history list.

        Returns:
            str: Assistant text response.

        Raises:
            LLMError: If server connection fails or response is invalid.
        """

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, None]:
        """
        Generates a streaming text completion response.

        Args:
            prompt: User message prompt.
            system_prompt: Optional system prompt context instructions.
            history: Optional sliding window chat history list.

        Yields:
            str: Streaming response tokens.

        Raises:
            LLMError: If server connection fails during stream generation.
        """

    @abstractmethod
    def close(self) -> None:
        """Closes reusable HTTP connection sessions and frees sockets."""


class OllamaLLMClient(LLMClient):
    """Concrete implementation of LLMClient communicating with local Ollama API."""

    def __init__(self, config: Config):
        """
        Initializes the Ollama HTTP client.

        Args:
            config: Loaded application configuration manager.
        """
        self.config = config
        self.host = config.llm.host.rstrip("/")
        self.model = config.llm.model
        self.timeout = config.llm.timeout
        self.temperature = config.llm.temperature
        self.max_tokens = config.llm.max_tokens

        # Reusable HTTP session
        self._session = requests.Session()
        logger.info("Ollama LLM Client initialized (host: %s, model: %s)", self.host, self.model)

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Compiles system context, chat history, and new prompt into message list."""
        messages: List[Dict[str, str]] = []

        # 1. Inject system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 2. Append history (roles: system, user, assistant)
        if history:
            messages.extend(history)

        # 3. Append current user prompt
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generates standard chat completion response."""
        url = f"{self.host}/api/chat"
        messages = self._build_messages(prompt, system_prompt, history)

        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            },
            "stream": False
        }

        try:
            logger.info("Sending request to Ollama (model: %s)", self.model)
            response = self._session.post(url, json=payload, timeout=self.timeout)

            if response.status_code != 200:
                raise LLMError(
                    f"Ollama server returned error status {response.status_code}: "
                    f"{response.text}"
                )

            data = response.json()
            assistant_message = data.get("message", {}).get("content", "").strip()
            logger.info("Ollama response received successfully.")
            return assistant_message

        except requests.RequestException as re:
            raise LLMError(f"Failed to communicate with local Ollama server: {str(re)}") from re
        except (ValueError, KeyError, TypeError) as parse_err:
            raise LLMError(f"Failed to parse Ollama JSON response: {str(parse_err)}") from parse_err

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, None]:
        """Streams chat completion response tokens."""
        url = f"{self.host}/api/chat"
        messages = self._build_messages(prompt, system_prompt, history)

        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            },
            "stream": True
        }

        try:
            logger.info("Sending streaming request to Ollama (model: %s)", self.model)
            response = self._session.post(url, json=payload, timeout=self.timeout, stream=True)

            if response.status_code != 200:
                raise LLMError(
                    f"Ollama server returned error status {response.status_code}: "
                    f"{response.text}"
                )

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done", False):
                        break

        except requests.RequestException as re:
            raise LLMError(f"Failed to stream response from local Ollama: {str(re)}") from re
        except (ValueError, KeyError, TypeError) as parse_err:
            raise LLMError(f"Stream JSON chunk parsing failed: {str(parse_err)}") from parse_err

    def close(self) -> None:
        """Closes HTTP session connection pools."""
        logger.info("Closing Ollama LLM Client HTTP session.")
        self._session.close()
