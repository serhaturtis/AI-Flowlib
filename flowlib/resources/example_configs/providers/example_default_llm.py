"""Example LlamaCpp provider configuration.

This file is automatically copied to ~/.flowlib/configs/ during initialization.
Alias bindings are handled separately in ~/.flowlib/configs/aliases.py.
Modify the settings below for your specific setup.
"""

from flowlib.resources.decorators.decorators import llm_config
from flowlib.config.required_resources import RequiredAlias


@llm_config("example-llamacpp-provider")
class ExampleLlamaCppProviderConfig:
    """Example LlamaCpp provider configuration.

    This configures the LlamaCpp provider infrastructure that can host multiple models.
    Individual models are defined separately using @model_config.
    This config can be assigned to roles like RequiredAlias.DEFAULT_LLM.value via role assignment.
    """

    provider_type = "llamacpp"
    settings = {
        # Pure infrastructure settings (LlamaCppSettings) - Provider concerns only
        "max_concurrent_models": 3,  # Maximum models loaded simultaneously
        # Provider reliability settings (from ProviderSettings)
        "timeout": 120,  # Request timeout in seconds
        "max_retries": 3,  # Maximum retry attempts
        "retry_delay": 1.0,  # Delay between retries
        "verbose": False,  # Enable verbose logging
    }
