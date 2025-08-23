"""Example configuration for default-embedding role.

This file shows how to configure an embedding model provider.
Copy this file to ~/.flowlib/active_configs/default_embedding.py and modify as needed.
"""

from flowlib.resources.decorators.decorators import embedding_config
from flowlib.resources.models.base import ResourceBase


@embedding_config("default-embedding")
class DefaultEmbeddingConfig(ResourceBase):
    """Example configuration for the default embedding provider.
    
    This configures the LlamaCpp embedding provider infrastructure that can host multiple embedding models.
    Individual embedding models are defined separately using @embedding_model_config.
    """
    
    def __init__(self, name: str, type: str, **kwargs):
        super().__init__(
            name=name,
            type=type,
            provider_type="llamacpp_embedding",
            settings={
                # Provider-level settings (LlamaCppEmbeddingProviderSettings) - Infrastructure only
                "n_threads": None,               # Number of threads (None = auto-detect)
                "n_batch": 512,                  # Default batch size for embedding processing
                "use_gpu": True,                 # Whether to enable GPU acceleration
                "n_gpu_layers": -1,              # Default GPU layers to offload (-1 = all)
                "verbose": False,                # Enable verbose logging from LlamaCpp
                "use_mlock": False,              # Use mlock to keep model in memory
                
                # Embedding provider settings
                "normalize": True,               # Whether to normalize embedding vectors by default
                
                # Provider reliability settings (from ProviderSettings)
                "timeout": 60,                   # Request timeout in seconds
                "max_retries": 3,                # Maximum retry attempts
                "retry_delay": 1.0,              # Delay between retries
            }
        )