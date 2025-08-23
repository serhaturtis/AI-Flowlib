"""Example LlamaCpp embedding provider configuration.

This file is automatically copied to ~/.flowlib/configs/ during initialization.
Role assignments are handled separately in ~/.flowlib/roles/assignments.py.
Modify the settings below for your specific setup.
"""

from flowlib.resources.decorators.decorators import embedding_config
from flowlib.resources.models.config_resource import EmbeddingConfigResource


@embedding_config("example-llamacpp-embedding-provider")
class ExampleLlamaCppEmbeddingProviderConfig(EmbeddingConfigResource):
    """Example LlamaCpp embedding provider configuration.
    
    This configures the LlamaCpp embedding provider infrastructure that can host multiple embedding models.
    Individual embedding models are defined separately using @embedding_model_config.
    This config can be assigned to roles like 'default-embedding' via role assignment.
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
                "batch_size": 32,                # Batch size for embedding processing
            }
        )