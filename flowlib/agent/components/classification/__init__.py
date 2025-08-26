"""
Message Classification module.

This module provides components for classifying user messages into conversation or task categories.
"""

from flowlib.agent.components.classification.flow import MessageClassifierFlow, MessageClassification, MessageClassifierInput
from flowlib.agent.components.classification import prompts  # Import to register prompts

__all__ = [
    "MessageClassifierFlow",
    "MessageClassification",
    "MessageClassifierInput"
] 