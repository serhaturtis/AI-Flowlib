"""Agent module for autonomous AI agents with memory, planning, and learning capabilities."""

from flowlib.agent.core.agent import AgentCore as Agent

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
    "Agent",
    "utils",
]