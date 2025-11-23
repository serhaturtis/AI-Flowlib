"""Example LlamaCpp Multimodal provider configuration.

This file is automatically copied to ~/.flowlib/configs/ during initialization.
Alias bindings are handled separately in ~/.flowlib/configs/aliases.py.
Modify the settings below for your specific setup.
"""

from flowlib.resources.decorators.decorators import multimodal_llm_provider_config


@multimodal_llm_provider_config("example-llamacpp-multimodal-provider")
class ExampleLlamaCppMultimodalProviderConfig:
    """Example LlamaCpp Multimodal provider configuration.

    This configures the LlamaCpp multimodal provider infrastructure that can host
    multiple vision-language models (LLaVA, Moondream, Llama 3.2 Vision, etc.).
    Individual models are defined separately using @multimodal_llm_config.
    This config can be assigned to roles like 'default-multimodal-llm' via role assignment.
    """

    provider_type = "llamacpp_multimodal"
    settings = {
        # Pure infrastructure settings (MultimodalLLMSettings) - Provider concerns only
        "max_concurrent_models": 2,  # Maximum multimodal models loaded simultaneously
        # Provider reliability settings (from ProviderSettings)
        "timeout": 180,  # Request timeout in seconds (multimodal generation can be slower)
        "max_retries": 3,  # Maximum retry attempts
        "retry_delay": 1.0,  # Delay between retries
        "verbose": False,  # Enable verbose logging
    }
