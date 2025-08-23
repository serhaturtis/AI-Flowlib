"""Example configuration for default-llm role.

This file shows how to configure a primary language model provider.
Copy this file to ~/.flowlib/active_configs/default_llm.py and modify as needed.
"""

from flowlib.resources.decorators.decorators import llm_config
from flowlib.resources.models.base import ResourceBase


@llm_config("default-llm")
class DefaultLLMConfig(ResourceBase):
    """Example configuration for the default LLM provider.
    
    This configures the LlamaCpp provider infrastructure that can host multiple models.
    Individual models are defined separately using @model_config.
    """
    
    def __init__(self, name: str, type: str, **kwargs):
        super().__init__(
            name=name,
            type=type,
            provider_type="llamacpp",
            settings={
                # Provider-level settings (LlamaCppSettings) - Infrastructure only
                "n_threads": 4,                    # Number of CPU threads for inference
                "n_batch": 512,                    # Batch size for processing optimization
                "use_gpu": True,                   # Enable GPU acceleration capability
                "n_gpu_layers": -1,                # Default GPU layers to offload (-1=all layers)
                "max_concurrent_models": 3,        # Maximum models loaded simultaneously
                
                # Provider reliability settings (from ProviderSettings)
                "timeout": 120,                    # Request timeout in seconds
                "max_retries": 3,                  # Maximum retry attempts
                "retry_delay": 1.0,                # Delay between retries
                "verbose": False,                  # Enable verbose logging
            }
        )