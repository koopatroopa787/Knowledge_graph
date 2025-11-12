"""Base LLM client interface."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model_name: str, temperature: float = 0.1,
                 max_tokens: int = 2000, timeout: int = 60):
        """
        Initialize the LLM client.

        Args:
            model_name: Name of the model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str,
                json_mode: bool = True) -> Optional[str]:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System prompt/instructions
            user_prompt: User prompt/query
            json_mode: Whether to request JSON output

        Returns:
            Generated response text or None on error
        """
        pass

    @abstractmethod
    def check_availability(self) -> bool:
        """
        Check if the LLM service is available.

        Returns:
            True if service is available, False otherwise
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
