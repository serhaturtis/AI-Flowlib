"""Example model configuration.

This file is automatically copied to ~/.flowlib/configs/ during initialization.
Role assignments are handled separately in ~/.flowlib/roles/assignments.py.
Modify the settings below for your specific setup.
"""

from flowlib.resources.decorators.decorators import model_config
from flowlib.providers.llm.models import LlamaModelConfig


@model_config("phi4-model", provider_type="llamacpp", config={
    "path": "/path/to/models/phi-4-q8_0.gguf",
    "model_type": "phi4",
    "n_ctx": 16384,
    "use_gpu": True,
    "n_gpu_layers": 32,
    "n_threads": 4,
    "n_batch": 512,
    "temperature": 0.7,
    "max_tokens": 4096,
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "chat_format": "chatml"
})
class Phi4ModelConfig(LlamaModelConfig):
    """Example configuration for a specific Phi-4 model.
    
    This creates a named model configuration that uses the "llamacpp" 
    provider type to load and run this specific model file.
    Model configs specify the provider_type, provider configs handle infrastructure.
    """
    pass