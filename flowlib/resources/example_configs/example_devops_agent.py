"""DevOps agent configuration.

An agent specialized in infrastructure and deployment tasks.
"""

from typing import Any

from flowlib.resources.decorators.decorators import agent_config
from flowlib.resources.models.agent_config_resource import AgentConfigResource


@agent_config("devops-agent-config")
class DevOpsAgentConfig(AgentConfigResource):
    """DevOps agent for infrastructure and deployment tasks."""

    def __init__(self, name: str, type: str, **kwargs: Any) -> None:
        super().__init__(
            name=name,
            type=type,
            persona="An experienced DevOps engineer focused on automation and reliability. "
                   "I prioritize infrastructure as code, CI/CD best practices, "
                   "monitoring, and system reliability.",
            profile_name="devops-agent-profile",
            model_name="default-model",
            llm_name="default-llm",
            temperature=0.5,
            max_iterations=12,
            enable_learning=True,
            verbose=False
        )
