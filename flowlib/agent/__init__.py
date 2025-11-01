"""Agent module for autonomous AI agents with memory, planning, and learning capabilities."""

# BaseAgent removed from init to avoid circular imports
# Import BaseAgent directly: from flowlib.agent.core.base_agent import BaseAgent

# New unified launcher system
from flowlib.agent.launcher import AgentLauncher

# Create utils alias for test compatibility
try:
    from flowlib.utils.agent.config import initialize_resources_from_config

    class AgentUtils:
        """Utility namespace for agent utilities."""

        initialize_resources_from_config = initialize_resources_from_config

    utils = AgentUtils()
except ImportError:
    raise RuntimeError("AgentUtils not available - required agent utilities missing") from None

__all__ = [
    "utils",
    "AgentLauncher",
]
