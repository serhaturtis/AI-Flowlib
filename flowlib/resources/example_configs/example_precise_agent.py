"""Precise agent configuration.

An agent optimized for accuracy and careful analysis.
"""

from typing import Any

from flowlib.resources.decorators.decorators import agent_config
from flowlib.resources.models.agent_config_resource import AgentConfigResource


@agent_config("precise-agent-config")
class PreciseAgentConfig(AgentConfigResource):
    """Precise agent for critical or security-sensitive tasks."""

    def __init__(self, name: str, type: str, **kwargs: Any) -> None:
        super().__init__(
            name=name,
            type=type,
            persona="A meticulous and detail-oriented engineer. "
                   "I double-check everything, prioritize correctness over speed, "
                   "and always consider edge cases and potential issues.",
            profile_name="restricted-agent-profile",  # More restricted access for safety
            model_name="default-model",
            llm_name="default-llm",
            temperature=0.3,  # Lower temperature for consistency
            max_iterations=20,  # More iterations for thoroughness
            enable_memory=True,
            enable_learning=False,  # Avoid learning patterns that might compromise precision
            enable_thinking=True,
            auto_decomposition=True,
            verbose=True  # Verbose to show all reasoning
        )