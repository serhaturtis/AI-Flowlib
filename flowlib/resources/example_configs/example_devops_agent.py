"""DevOps agent configuration.

An agent specialized in infrastructure and deployment tasks.
"""

from flowlib.resources.decorators.decorators import agent_config
from flowlib.config.required_resources import RequiredAlias


@agent_config("devops-agent-config")
class DevOpsAgentConfig:
    """DevOps agent for infrastructure and deployment tasks."""

    persona = (
        "An experienced DevOps engineer focused on automation and reliability. "
        "I prioritize infrastructure as code, CI/CD best practices, "
        "monitoring, and system reliability."
    )
    allowed_tool_categories = ["generic", "devops"]
    model_name = RequiredAlias.DEFAULT_MODEL.value
    llm_name = RequiredAlias.DEFAULT_LLM.value
    temperature = 0.5
    max_iterations = 12
    enable_learning = True
    verbose = False
