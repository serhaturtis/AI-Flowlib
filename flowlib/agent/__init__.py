"""Agent module for autonomous AI agents with memory, planning, and learning capabilities."""

from flowlib.agent.core.base_agent import BaseAgent

# Import debriefing components to trigger decorator registration
from flowlib.agent.components.task.debriefing import flows, prompts

# Create utils alias for test compatibility
try:
    from flowlib.utils.agent import model_config
    class AgentUtils:
        """Utility namespace for agent utilities."""
        model_config = model_config
    utils = AgentUtils()
except ImportError:
    utils = None

__all__ = [
    "BaseAgent",
    "utils",
]