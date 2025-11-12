"""OpenAI LLM client implementation."""

import logging
from typing import Optional
from openai import OpenAI
from .base import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI API client for LLM operations."""

    def __init__(self, model_name: str, api_key: str,
                 temperature: float = 0.1, max_tokens: int = 2000,
                 timeout: int = 60):
        """
        Initialize OpenAI client.

        Args:
            model_name: OpenAI model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
            api_key: OpenAI API key
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        super().__init__(model_name, temperature, max_tokens, timeout)
        self.client = OpenAI(api_key=api_key, timeout=timeout)

    def generate(self, system_prompt: str, user_prompt: str,
                json_mode: bool = True) -> Optional[str]:
        """Generate a response using OpenAI API."""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

            if json_mode and "gpt-4" in self.model_name or "gpt-3.5" in self.model_name:
                kwargs["response_format"] = {"type": "json_object"}
                # Add JSON instruction to system prompt if not already present
                if "json" not in system_prompt.lower():
                    messages[0]["content"] += "\n\nRespond with valid JSON only."

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return None

    def check_availability(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI service unavailable: {str(e)}")
            return False
