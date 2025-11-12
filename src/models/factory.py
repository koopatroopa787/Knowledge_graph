"""Factory for creating LLM clients."""

import logging
from typing import Optional
from .base import BaseLLMClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating LLM clients based on provider."""

    @staticmethod
    def create_client(provider: str, model_name: str,
                     api_key: Optional[str] = None,
                     host: Optional[str] = None,
                     temperature: float = 0.1,
                     max_tokens: int = 2000,
                     timeout: int = 60) -> BaseLLMClient:
        """
        Create an LLM client based on the provider.

        Args:
            provider: Provider name ('openai', 'anthropic', 'ollama')
            model_name: Model name to use
            api_key: API key (for OpenAI and Anthropic)
            host: Host URL (for Ollama)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds

        Returns:
            LLM client instance

        Raises:
            ValueError: If provider is unsupported or configuration is invalid
        """
        provider = provider.lower()

        if provider == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required")
            return OpenAIClient(
                model_name=model_name,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )

        elif provider == "anthropic":
            if not api_key:
                raise ValueError("Anthropic API key is required")
            return AnthropicClient(
                model_name=model_name,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )

        elif provider == "ollama":
            host = host or "http://localhost:11434"
            return OllamaClient(
                model_name=model_name,
                host=host,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )

        else:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers: openai, anthropic, ollama"
            )

    @staticmethod
    def from_config(config) -> BaseLLMClient:
        """
        Create an LLM client from a Config object.

        Args:
            config: Config object with LLM settings

        Returns:
            LLM client instance
        """
        provider = config.llm_provider
        model_name = config.model_name

        kwargs = {
            'provider': provider,
            'model_name': model_name,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'timeout': config.timeout,
        }

        if provider in ['openai', 'anthropic']:
            kwargs['api_key'] = config.get_api_key(provider)
        elif provider == 'ollama':
            kwargs['host'] = config.get_ollama_host()

        return LLMFactory.create_client(**kwargs)
