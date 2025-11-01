"""Example LlamaCpp embedding provider configuration.

This file is automatically copied to ~/.flowlib/configs/ during initialization.
Role assignments are handled separately in ~/.flowlib/roles/assignments.py.
Modify the settings below for your specific setup.
"""

from flowlib.resources.decorators.decorators import embedding_config


@embedding_config(
    "example-llamacpp-embedding-provider",
    provider_type="llamacpp_embedding",
    config={
        "n_ctx": 512,
        "n_threads": None,
        "n_batch": 512,
        "use_gpu": True,
        "n_gpu_layers": -1,
        "verbose": False,
        "use_mlock": False,
        "normalize": True,
        "batch_size": 32,
    },
)
class ExampleLlamaCppEmbeddingProviderConfig:
    """Example LlamaCpp embedding provider configuration.

    This configures the LlamaCpp embedding provider infrastructure that can host multiple embedding models.
    Individual embedding models are defined separately using @embedding_model_config.
    This config can be assigned to roles like 'default-embedding' via role assignment.
    """

    pass
