"""
LLM Subsystem Package for ZovaAI.
Contains interfaces, concrete local Ollama clients, and service wrappers.
"""

from src.llm.ollama_client import LLMClient, OllamaLLMClient
from src.llm.service import LLMService

__all__ = [
    "LLMClient",
    "OllamaLLMClient",
    "LLMService"
]
