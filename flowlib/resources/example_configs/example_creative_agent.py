"""Creative agent configuration.

An agent optimized for brainstorming and innovative solutions.
"""

from flowlib.resources.decorators.decorators import agent_config


@agent_config("creative-agent-config")
class CreativeAgentConfig:
    """Creative agent for brainstorming and innovative solutions."""

    persona = (
        "An innovative and creative problem solver. "
        "I think outside the box, suggest novel approaches, "
        "and help brainstorm unique solutions to challenges."
    )
    profile_name = "default-agent-profile"
    model_name = "default-model"
    llm_name = "default-llm"
    temperature = 0.9  # Higher temperature for creativity
    max_iterations = 15
    enable_learning = True
    verbose = False
