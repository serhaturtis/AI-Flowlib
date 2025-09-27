"""Template for creating custom agent configurations.

Copy this file and modify it to create your own agent configurations.
Place your custom agent configs in ~/.flowlib/configs/
"""

from typing import Any

from flowlib.resources.decorators.decorators import agent_config
from flowlib.resources.models.agent_config_resource import AgentConfigResource


# Example: Custom agent for code review
@agent_config("code-review-agent")
class CodeReviewAgentConfig(AgentConfigResource):
    """Agent specialized in code review and quality analysis."""

    def __init__(self, name: str, type: str, **kwargs: Any) -> None:
        super().__init__(
            name=name,
            type=type,
            persona="A senior software architect specializing in code review. "
                   "I focus on code quality, design patterns, security vulnerabilities, "
                   "and provide constructive feedback with specific improvement suggestions.",
            profile_name="qa-automation-agent-profile",  # QA engineer role for testing tools
            model_name="default-model",
            llm_name="default-llm",
            temperature=0.4,  # Lower temperature for consistent analysis
            max_iterations=15,
            enable_memory=True,
            enable_learning=True,
            enable_thinking=True,
            auto_decomposition=True,
            verbose=False,
            additional_settings={
                "check_style": True,
                "check_security": True,
                "suggest_improvements": True
            }
        )


# Example: Custom agent with specific model
"""
@agent_config("my-gpt4-agent")
class MyGPT4AgentConfig(AgentConfigResource):
    def __init__(self, name: str, type: str, **kwargs: Any) -> None:
        super().__init__(
            name=name,
            type=type,
            persona="Your custom persona here",
            profile_name="software-engineer-profile",
            model_name="gpt4-model",  # Reference your GPT-4 model config
            llm_name="openai-llm",     # Reference your OpenAI provider config
            temperature=0.7,
            # ... other settings
        )
"""

# To use your custom agent:
# 1. Save this file in ~/.flowlib/configs/my_custom_agents.py
# 2. Add role assignment in ~/.flowlib/roles/assignments.py:
#    role_manager.assign_role("my-agent", "code-review-agent")
# 3. Run the REPL:
#    ./flowlib/apps/run_repl.py --agent-config my-agent