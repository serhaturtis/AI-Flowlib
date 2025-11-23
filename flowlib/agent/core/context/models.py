"""Unified context models for agent operations.

This module defines the single source of truth for all agent context,
replacing fragmented context types with unified models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from flowlib.agent.components.task.core.todo import TodoItem
from flowlib.agent.models.conversation import ConversationMessage
from flowlib.core.models import MutableStrictBaseModel, StrictBaseModel


class ContextManagerConfig(StrictBaseModel):
    """Configuration for AgentContextManager.

    Lives with context module following Single Source of Truth principle.
    """

    auto_compaction_threshold: int = Field(
        default=180000, gt=0, description="Token threshold for auto-compaction"
    )
    session_persistence: bool = Field(
        default=True, description="Whether to persist session context"
    )
    learning_enabled: bool = Field(default=True, description="Whether to learn from user patterns")
    workspace_scanning: bool = Field(
        default=True, description="Whether to scan workspace for knowledge"
    )
    max_conversation_history: int = Field(
        default=100, gt=0, description="Maximum conversation history to maintain"
    )
    pattern_learning_threshold: int = Field(
        default=3, gt=0, description="Minimum success count to consider pattern learned"
    )


class UserProfile(StrictBaseModel):
    """Learned user patterns and preferences."""

    preferred_tools: list[str] = Field(default_factory=list, description="User's preferred tools")
    coding_style: dict[str, Any] = Field(
        default_factory=dict, description="User's coding style preferences"
    )
    communication_style: str = Field(default="direct", description="User's communication style")
    domain_expertise: list[str] = Field(
        default_factory=list, description="User's areas of expertise"
    )


class WorkspaceKnowledge(StrictBaseModel):
    """Knowledge about the current project/workspace."""

    project_type: str | None = Field(default=None, description="Detected project type")
    languages: list[str] = Field(
        default_factory=list, description="Programming languages in project"
    )
    dependencies: list[str] = Field(default_factory=list, description="Project dependencies")
    common_patterns: list[str] = Field(default_factory=list, description="Common code patterns")
    file_structure: dict[str, Any] = Field(
        default_factory=dict, description="Project file structure overview"
    )


class SuccessfulPattern(MutableStrictBaseModel):
    """Pattern that has been successful in the past."""

    pattern_type: str = Field(..., description="Type of successful pattern")
    description: str = Field(..., description="Pattern description")
    success_count: int = Field(default=1, description="Number of times this pattern succeeded")
    last_used: datetime = Field(
        default_factory=datetime.now, description="Last time pattern was used"
    )


class RecoveryStrategy(StrictBaseModel):
    """Strategy for recovering from errors."""

    error_type: str = Field(..., description="Type of error this strategy addresses")
    strategy: str = Field(..., description="Recovery strategy description")
    success_rate: float = Field(default=0.0, description="Success rate of this strategy")
    usage_count: int = Field(default=0, description="Number of times strategy was used")


class SessionContext(MutableStrictBaseModel):
    """Session-wide persistent context."""

    # Session identification
    session_id: str = Field(..., description="Unique session identifier")
    user_id: str | None = Field(default=None, description="User identifier")
    agent_name: str = Field(..., description="Agent name")
    allowed_tool_categories: list[str] = Field(
        default_factory=list, description="Tool categories the agent may use"
    )
    agent_persona: str = Field(..., description="Agent persona")
    working_directory: str = Field(..., description="Current working directory")

    # Conversation state
    current_message: str = Field(..., description="Current user message")
    conversation_history: list[ConversationMessage] = Field(
        default_factory=list, description="Full conversation history"
    )

    # Shared context for collaboration
    shared_context: dict[str, Any] = Field(
        default_factory=dict, description="Shared context between agents"
    )
    collaborating_agents: list[str] = Field(
        default_factory=list, description="List of collaborating agent names"
    )
    conversation_summary: str | None = Field(
        default=None, description="Auto-compacted conversation summary"
    )

    # Learning state
    user_profile: UserProfile = Field(
        default_factory=UserProfile, description="Learned user patterns"
    )
    workspace_knowledge: WorkspaceKnowledge = Field(
        default_factory=WorkspaceKnowledge, description="Project understanding"
    )


class TaskContext(MutableStrictBaseModel):
    """Current task execution context."""

    description: str = Field(..., description="Task description")
    cycle: int = Field(default=1, description="Current execution cycle")
    todos: list[TodoItem] = Field(default_factory=list, description="Current TODOs")
    execution_results: list[dict[str, Any]] = Field(
        default_factory=list, description="Results from previous cycles"
    )
    started_at: datetime = Field(default_factory=datetime.now, description="Task start time")


class ComponentContext(StrictBaseModel):
    """Component-specific execution context."""

    component_type: Literal[
        "task_generation",
        "task_thinking",
        "task_decomposition",
        "task_execution",
        "task_debriefing",
        "task_planning",
        "task_evaluation",  # New Plan-Execute-Evaluate components
    ] = Field(..., description="Current component type")
    component_config: dict[str, Any] = Field(
        default_factory=dict, description="Component-specific configuration"
    )

    # Tool and execution settings
    execution_timeout: int | None = Field(
        default=None, description="Execution timeout in seconds"
    )
    retry_policy: str = Field(default="simple", description="Retry policy for failed operations")


class LearningContext(MutableStrictBaseModel):
    """Learning and adaptation context."""

    successful_patterns: list[SuccessfulPattern] = Field(
        default_factory=list, description="Previously successful patterns"
    )
    user_preferences: dict[str, Any] = Field(
        default_factory=dict, description="Learned user preferences"
    )
    error_recovery_strategies: list[RecoveryStrategy] = Field(
        default_factory=list, description="Error recovery strategies"
    )
    total_executions: int = Field(default=0, description="Total number of executions")
    successful_executions: int = Field(default=0, description="Number of successful executions")


class ExecutionContext(StrictBaseModel):
    """THE unified context model for all agent operations."""

    session: SessionContext = Field(..., description="Session-wide persistent context")
    task: TaskContext = Field(..., description="Current task context")
    component: ComponentContext = Field(..., description="Component-specific context")
    learning: LearningContext = Field(..., description="Learning and adaptation context")

    @property
    def token_count(self) -> int:
        """Estimate token count for auto-compaction."""
        # Simple estimation based on string lengths
        session_tokens = len(str(self.session.conversation_history)) // 4
        task_tokens = len(self.task.description) // 4
        summary_tokens = len(self.session.conversation_summary or "") // 4

        return session_tokens + task_tokens + summary_tokens

    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        if self.learning.total_executions == 0:
            return 0.0
        return self.learning.successful_executions / self.learning.total_executions
