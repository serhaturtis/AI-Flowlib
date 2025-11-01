"""Default agent configuration.

This file is automatically copied to ~/.flowlib/configs/ during initialization.
Modify the configuration below for your default agent setup.
"""

from flowlib.resources.decorators.decorators import agent_config


@agent_config("default-agent-config")
class DefaultAgentConfig:
    """Default agent configuration for general development tasks."""

    persona = (
        "A helpful and knowledgeable software engineering assistant. "
        "I write clean, well-documented code and follow best practices. "
        "I'm thorough in testing and careful about security."
    )
    profile_name = "default-agent-profile"  # Uses software_engineer role
    model_name = "default-model"
    llm_name = "default-llm"
    temperature = 0.7
    max_iterations = 10
    enable_learning = True
    verbose = False
