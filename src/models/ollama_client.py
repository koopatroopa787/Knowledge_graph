"""Ollama LLM client implementation."""

import json
import logging
import requests
from typing import Optional
from .base import BaseLLMClient

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):
    """Ollama API client for LLM operations."""

    def __init__(self, model_name: str, host: str = "http://localhost:11434",
                 temperature: float = 0.1, max_tokens: int = 2000,
                 timeout: int = 60):
        """
        Initialize Ollama client.

        Args:
            model_name: Ollama model name (e.g., 'llama3', 'mistral')
            host: Ollama server host URL
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        super().__init__(model_name, temperature, max_tokens, timeout)
        self.host = host.rstrip('/')

    def generate(self, system_prompt: str, user_prompt: str,
                json_mode: bool = True) -> Optional[str]:
        """Generate a response using Ollama API."""
        try:
            url = f"{self.host}/api/generate"

            # Add JSON instruction if json_mode is enabled
            if json_mode and "json" not in system_prompt.lower():
                system_prompt += "\n\nRespond with valid JSON only."

            payload = {
                "model": self.model_name,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            }

            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return None

    def check_availability(self) -> bool:
        """Check if Ollama service is available."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException:
            logger.error(f"Ollama service unavailable at {self.host}")
            return False

    def list_models(self):
        """List available Ollama models."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            return response.json().get('models', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list models: {str(e)}")
            return []
