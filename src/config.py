"""Configuration management for Knowledge Graph Generator."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration manager for the application."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _validate_config(self):
        """Validate configuration settings."""
        required_keys = ['llm', 'document_processing', 'graph', 'output']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration section: {key}")

    @property
    def llm_provider(self) -> str:
        """Get the LLM provider."""
        return self.config['llm']['provider']

    @property
    def model_name(self) -> str:
        """Get the model name for the current provider."""
        provider = self.llm_provider
        return self.config['llm']['models'].get(provider)

    @property
    def temperature(self) -> float:
        """Get the temperature setting."""
        return self.config['llm'].get('temperature', 0.1)

    @property
    def max_tokens(self) -> int:
        """Get the max tokens setting."""
        return self.config['llm'].get('max_tokens', 2000)

    @property
    def timeout(self) -> int:
        """Get the timeout setting."""
        return self.config['llm'].get('timeout', 60)

    @property
    def chunk_size(self) -> int:
        """Get the chunk size for document processing."""
        return self.config['document_processing']['chunk_size']

    @property
    def chunk_overlap(self) -> int:
        """Get the chunk overlap for document processing."""
        return self.config['document_processing']['chunk_overlap']

    @property
    def batch_size(self) -> int:
        """Get the batch size for processing."""
        return self.config['document_processing']['batch_size']

    @property
    def delay_between_batches(self) -> int:
        """Get the delay between batches."""
        return self.config['document_processing']['delay_between_batches']

    @property
    def min_importance(self) -> int:
        """Get the minimum importance threshold."""
        return self.config['graph']['min_importance']

    @property
    def output_dir(self) -> Path:
        """Get the output directory."""
        return Path(self.config['output']['directory'])

    @property
    def save_intermediate(self) -> bool:
        """Check if intermediate files should be saved."""
        return self.config['output'].get('save_intermediate', True)

    def get_api_key(self, provider: str = None) -> str:
        """
        Get API key for the specified provider.

        Args:
            provider: LLM provider name (defaults to current provider)

        Returns:
            API key from environment variable
        """
        if provider is None:
            provider = self.llm_provider

        key_map = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
        }

        env_var = key_map.get(provider)
        if env_var:
            api_key = os.getenv(env_var)
            if not api_key:
                raise ValueError(
                    f"API key not found for {provider}. "
                    f"Please set {env_var} in your .env file"
                )
            return api_key
        return None

    def get_ollama_host(self) -> str:
        """Get Ollama host URL."""
        return os.getenv('OLLAMA_HOST', 'http://localhost:11434')

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key path.

        Args:
            key: Dot-separated key path (e.g., 'graph.visualization.height')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
