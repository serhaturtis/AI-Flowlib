"""Shared context models for task components.

This module contains context models with no dependencies on other components.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from flowlib.core.models import StrictBaseModel


class RequestContext(StrictBaseModel):
    """Context for task decomposition and execution requests."""

    session_id: str | None = Field(default=None, description="Agent session ID")
    user_id: str | None = Field(default=None, description="User identifier")
    agent_name: str | None = Field(default=None, description="Agent name")
    agent_role: str | None = Field(
        default=None, description="Agent's role for tool access control"
    )
    previous_messages: list[Any] = Field(
        default_factory=list, description="Previous conversation messages"
    )
    working_directory: str = Field(default=".", description="Working directory")
    agent_persona: str | None = Field(default=None, description="Agent's persona/personality")
    memory_context: str | None = Field(default=None, description="Memory context string")
