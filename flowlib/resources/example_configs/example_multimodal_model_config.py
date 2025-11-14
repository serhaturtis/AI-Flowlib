"""Example multimodal LLM model configuration.

This file is automatically copied to ~/.flowlib/configs/ during initialization.
Role assignments are handled separately in ~/.flowlib/roles/assignments.py.
Modify the settings below for your specific setup.
"""

from flowlib.resources.decorators.decorators import multimodal_llm_config


@multimodal_llm_config(
    "llava-1.5-7b-model",
    provider_type="llamacpp_multimodal",
    config={
        # Model files (both required for multimodal)
        "path": "/path/to/models/llava-v1.5-7b-q8_0.gguf",
        "clip_model_path": "/path/to/models/mmproj-model-f16.gguf",
        # Context and GPU settings
        "n_ctx": 4096,
        "use_gpu": True,
        "n_gpu_layers": 32,
        "n_threads": 4,
        "n_batch": 512,
        # Generation parameters
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p": 0.95,
        "top_k": 40,
        "repeat_penalty": 1.1,
        # Chat format for multimodal models
        "chat_format": "llava-1-5",  # Options: llava-1-5, moondream, llama-3-vision
    },
)
class LLaVA15ModelConfig:
    """Example configuration for LLaVA 1.5 7B multimodal model.

    LLaVA (Large Language and Vision Assistant) is a vision-language model
    that can understand and respond to images along with text prompts.

    This creates a named model configuration that uses the "llamacpp_multimodal"
    provider type to load and run this specific multimodal model.
    Model configs specify the provider_type, provider configs handle infrastructure.

    Requirements:
    - Model file (.gguf format)
    - CLIP projection model file (mmproj-*.gguf)
    - Both files must be compatible with each other
    """

    pass


@multimodal_llm_config(
    "moondream-model",
    provider_type="llamacpp_multimodal",
    config={
        # Model files
        "path": "/path/to/models/moondream2-text-model-f16.gguf",
        "clip_model_path": "/path/to/models/moondream2-mmproj-f16.gguf",
        # Context and GPU settings
        "n_ctx": 2048,
        "use_gpu": True,
        "n_gpu_layers": 24,
        "n_threads": 4,
        "n_batch": 256,
        # Generation parameters
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.95,
        "top_k": 40,
        "repeat_penalty": 1.1,
        # Chat format
        "chat_format": "moondream",
    },
)
class MoondreamModelConfig:
    """Example configuration for Moondream multimodal model.

    Moondream is a compact vision-language model optimized for efficiency.
    Good for image understanding tasks with lower resource requirements.
    """

    pass
