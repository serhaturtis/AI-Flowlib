"""Default agent configuration.

This file is automatically copied to ~/.flowlib/configs/ during initialization.
Modify the configuration below for your default agent setup.
"""

from flowlib.resources.decorators.decorators import agent_config
from flowlib.config.required_resources import RequiredAlias


@agent_config("default-agent-config")
class DefaultAgentConfig:
    """Default agent configuration for general development tasks."""

    persona = (
        "A helpful and knowledgeable software engineering assistant. "
        "I write clean, well-documented code and follow best practices. "
        "I'm thorough in testing and careful about security."
    )
    allowed_tool_categories = ["generic", "software"]
    model_name = RequiredAlias.DEFAULT_MODEL.value
    llm_name = RequiredAlias.DEFAULT_LLM.value
    temperature = 0.7
    max_iterations = 10
    enable_learning = True
    verbose = False
