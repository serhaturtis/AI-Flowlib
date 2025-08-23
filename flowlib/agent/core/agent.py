"""
Agent core implementation.

This module provides the agent interface using the decomposed orchestrator architecture.
Following the principle of single source of truth with no backwards compatibility.
"""

# Single source of truth - import the orchestrator as AgentCore
from flowlib.agent.core.orchestrator import AgentOrchestrator as AgentCore

__all__ = ["AgentCore"]