"""Precise agent configuration.

An agent optimized for accuracy and careful analysis.
"""

from flowlib.resources.decorators.decorators import agent_config
from flowlib.config.required_resources import RequiredAlias


@agent_config("precise-agent-config")
class PreciseAgentConfig:
    """Precise agent for critical or security-sensitive tasks."""

    persona = (
        "A meticulous and detail-oriented engineer. "
        "I double-check everything, prioritize correctness over speed, "
        "and always consider edge cases and potential issues."
    )
    allowed_tool_categories = ["generic"]
    model_name = RequiredAlias.DEFAULT_MODEL.value
    llm_name = RequiredAlias.DEFAULT_LLM.value
    temperature = 0.3  # Lower temperature for consistency
    max_iterations = 20  # More iterations for thoroughness
    enable_learning = False  # Avoid learning patterns that might compromise precision
    verbose = True  # Verbose to show all reasoning
