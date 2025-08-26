"""Agent module for autonomous AI agents with memory, planning, and learning capabilities."""

from flowlib.agent.core.base_agent import BaseAgent
from flowlib.agent.agents.dual_path import DualPathAgent

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
    "DualPathAgent", 
    "utils",
]