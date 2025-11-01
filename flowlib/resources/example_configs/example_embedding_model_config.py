"""Example embedding model configuration.

This file is automatically copied to ~/.flowlib/configs/ during initialization.
Role assignments are handled separately in ~/.flowlib/roles/assignments.py.
Modify the settings below for your specific setup.
"""

from flowlib.resources.decorators.decorators import embedding_config


@embedding_config(
    "example-embedding-model",
    provider_type="llamacpp_embedding",
    config={
        "path": "/path/to/embedding_models/bge-m3-q8_0.gguf",
        "model_name": "bge-m3",
        "dimensions": 1024,
        "max_length": 8192,
        "normalize": True,
    },
)
class BGEM3EmbeddingModelConfig:
    """Example configuration for BGE-M3 embedding model.

    This creates a named embedding model configuration that uses the "llamacpp_embedding"
    provider type to load and run this specific embedding model file.
    Embedding configs specify the provider_type, provider configs handle infrastructure.
    """

    pass
