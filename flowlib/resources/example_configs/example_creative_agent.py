"""Creative agent configuration.

An agent optimized for brainstorming and innovative solutions.
"""

from flowlib.resources.decorators.decorators import agent_config
from flowlib.config.required_resources import RequiredAlias


@agent_config("creative-agent-config")
class CreativeAgentConfig:
    """Creative agent for brainstorming and innovative solutions."""

    persona = (
        "An innovative and creative problem solver. "
        "I think outside the box, suggest novel approaches, "
        "and help brainstorm unique solutions to challenges."
    )
    allowed_tool_categories = ["generic"]
    model_name = RequiredAlias.DEFAULT_MODEL.value
    llm_name = RequiredAlias.DEFAULT_LLM.value
    temperature = 0.9  # Higher temperature for creativity
    max_iterations = 15
    enable_learning = True
    verbose = False
