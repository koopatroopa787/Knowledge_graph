"""Anthropic LLM client implementation."""

import logging
from typing import Optional
from anthropic import Anthropic
from .base import BaseLLMClient

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """Anthropic API client for LLM operations."""

    def __init__(self, model_name: str, api_key: str,
                 temperature: float = 0.1, max_tokens: int = 2000,
                 timeout: int = 60):
        """
        Initialize Anthropic client.

        Args:
            model_name: Anthropic model name (e.g., 'claude-3-5-sonnet-20241022')
            api_key: Anthropic API key
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        super().__init__(model_name, temperature, max_tokens, timeout)
        self.client = Anthropic(api_key=api_key, timeout=timeout)

    def generate(self, system_prompt: str, user_prompt: str,
                json_mode: bool = True) -> Optional[str]:
        """Generate a response using Anthropic API."""
        try:
            # Add JSON instruction if json_mode is enabled
            if json_mode and "json" not in system_prompt.lower():
                system_prompt += "\n\nRespond with valid JSON only."

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            return None

    def check_availability(self) -> bool:
        """Check if Anthropic API is available."""
        try:
            # Try a minimal request to check availability
            self.client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception as e:
            logger.error(f"Anthropic service unavailable: {str(e)}")
            return False
