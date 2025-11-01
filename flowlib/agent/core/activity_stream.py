"""Real-time activity streaming for agent operations."""

import json
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any, TypedDict


class ActivityType(Enum):
    """Types of agent activities."""

    PLANNING = "🧠 Planning"
    MEMORY_RETRIEVAL = "🔍 Memory"
    MEMORY_STORE = "💾 Memory Store"
    FLOW_SELECTION = "🎯 Flow Selection"
    PROMPT_SELECTION = "📝 Prompt"
    LLM_CALL = "🤖 LLM"
    REFLECTION = "🤔 Reflection"
    TODO_CREATE = "📋 TODO Create"
    TODO_UPDATE = "✅ TODO Update"
    TODO_STATUS = "📊 TODO Status"
    LEARNING = "🎓 Learning"
    ERROR = "❌ Error"
    EXECUTION = "⚡ Execution"
    CONTEXT = "🌐 Context"
    DECISION = "💭 Decision"


class ActivityEntry(TypedDict):
    """Type definition for activity buffer entries."""

    timestamp: datetime
    type: ActivityType
    message: str
    details: dict[str, Any] | None


class ActivityStream:
    """Manages real-time activity streaming for agent operations."""

    def __init__(self, output_handler: Callable[[str], None] | None = None):
        """Initialize activity stream.

        Args:
            output_handler: Callback to handle activity output (e.g., print to console)
        """
        self.output_handler = output_handler or print
        self.enabled = True
        self.verbose = True
        self.activity_buffer: list[ActivityEntry] = []
        self.current_indent = 0

    def set_output_handler(self, handler: Callable[[str], None]) -> None:
        """Set the output handler for activities."""
        self.output_handler = handler

    def enable(self) -> None:
        """Enable activity streaming."""
        self.enabled = True

    def disable(self) -> None:
        """Disable activity streaming."""
        self.enabled = False

    def set_verbose(self, verbose: bool) -> None:
        """Set verbose mode for detailed output."""
        self.verbose = verbose

    def _format_activity(
        self, activity_type: ActivityType, message: str, details: dict[str, Any] | None = None
    ) -> str:
        """Format an activity message.

        Args:
            activity_type: Type of activity
            message: Main activity message
            details: Optional details dictionary

        Returns:
            Formatted activity string
        """
        indent = "  " * self.current_indent
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Main message
        lines = [f"{indent}[{timestamp}] {activity_type.value}: {message}"]

        # Add details if provided and verbose mode is on
        if details and self.verbose:
            for key, value in details.items():
                # Format value based on type
                if isinstance(value, dict):
                    value_str = json.dumps(value, indent=2)
                    # Indent each line of JSON
                    value_lines = value_str.split("\n")
                    value_str = "\n".join(f"{indent}    {line}" for line in value_lines)
                elif isinstance(value, list) and len(value) > 3:
                    value_str = f"[{len(value)} items]"
                else:
                    value_str = str(value)

                lines.append(f"{indent}  → {key}: {value_str}")

        return "\n".join(lines)

    def stream(
        self, activity_type: ActivityType, message: str, details: dict[str, Any] | None = None
    ) -> None:
        """Stream an activity in real-time.

        Args:
            activity_type: Type of activity
            message: Activity message
            details: Optional activity details
        """
        if not self.enabled:
            return

        activity = self._format_activity(activity_type, message, details)

        # Buffer for later retrieval if needed
        self.activity_buffer.append(
            {
                "timestamp": datetime.now(),
                "type": activity_type,
                "message": message,
                "details": details,
            }
        )

        # Output immediately
        self.output_handler(activity)

    def start_section(self, title: str) -> None:
        """Start a new activity section with indentation."""
        if self.enabled:
            self.output_handler(f"\n{'═' * 50}")
            self.output_handler(f"▶ {title}")
            self.output_handler(f"{'─' * 50}")
            self.current_indent += 1

    def end_section(self) -> None:
        """End the current activity section."""
        if self.enabled and self.current_indent > 0:
            self.current_indent -= 1
            self.output_handler(f"{'─' * 50}\n")

    # Convenience methods for common activities

    def planning(self, message: str, **details: Any) -> None:
        """Stream a planning activity."""
        self.stream(ActivityType.PLANNING, message, details)

    def memory_retrieval(
        self, query: str, results: list[Any] | None = None, **details: Any
    ) -> None:
        """Stream a memory retrieval activity."""
        detail_dict: dict[str, Any] = {"query": query}
        if results:
            detail_dict["found"] = len(results)
            if self.verbose and results:
                detail_dict["samples"] = results[:2]  # Show first 2
        detail_dict.update(details)
        self.stream(ActivityType.MEMORY_RETRIEVAL, f"Retrieving: {query[:50]}...", detail_dict)

    def memory_store(self, key: str, value: Any, **details: Any) -> None:
        """Stream a memory store activity."""
        detail_dict: dict[str, Any] = {"key": key, "type": type(value).__name__}
        detail_dict.update(details)
        self.stream(ActivityType.MEMORY_STORE, f"Storing: {key}", detail_dict)

    def flow_selection(
        self, selected: str, reasoning: str, alternatives: list[Any] | None = None
    ) -> None:
        """Stream a flow selection activity."""
        details: dict[str, Any] = {
            "selected": selected,
            "reasoning": reasoning[:100] + "..." if len(reasoning) > 100 else reasoning,
        }
        if alternatives:
            details["alternatives"] = alternatives
        self.stream(ActivityType.FLOW_SELECTION, f"Selected flow: {selected}", details)

    def prompt_selection(
        self, prompt_name: str, variables: dict[str, Any] | None = None
    ) -> None:
        """Stream a prompt selection activity."""
        details: dict[str, Any] = {"prompt": prompt_name}
        if variables:
            details["variables"] = list(variables.keys())
        self.stream(ActivityType.PROMPT_SELECTION, f"Using prompt: {prompt_name}", details)

    def llm_call(self, model: str, prompt_preview: str, **details: Any) -> None:
        """Stream an LLM call activity."""
        detail_dict: dict[str, Any] = {
            "model": model,
            "preview": prompt_preview[:100] + "..."
            if len(prompt_preview) > 100
            else prompt_preview,
        }
        detail_dict.update(details)
        self.stream(ActivityType.LLM_CALL, f"Calling {model}", detail_dict)

    def reflection(self, message: str, progress: int | None = None, **details: Any) -> None:
        """Stream a reflection activity."""
        detail_dict: dict[str, Any] = {"reflection": message}
        if progress is not None:
            detail_dict["progress"] = f"{progress}%"
        detail_dict.update(details)
        self.stream(ActivityType.REFLECTION, "Reflecting on execution", detail_dict)

    def todo_create(self, todo_content: str, priority: str = "MEDIUM", **details: Any) -> None:
        """Stream a TODO creation activity."""
        detail_dict: dict[str, Any] = {"content": todo_content, "priority": priority}
        detail_dict.update(details)
        self.stream(ActivityType.TODO_CREATE, f"Creating TODO: {todo_content[:50]}...", detail_dict)

    def todo_update(self, todo_id: str, status: str, **details: Any) -> None:
        """Stream a TODO update activity."""
        detail_dict: dict[str, Any] = {"id": todo_id, "status": status}
        detail_dict.update(details)
        self.stream(ActivityType.TODO_UPDATE, f"TODO {todo_id[:8]} → {status}", detail_dict)

    def todo_status(self, total: int, completed: int, in_progress: int) -> None:
        """Stream TODO status summary."""
        self.stream(
            ActivityType.TODO_STATUS,
            f"TODOs: {completed}/{total} completed, {in_progress} in progress",
        )

    def learning(self, what: str, entities: list[Any] | None = None, **details: Any) -> None:
        """Stream a learning activity."""
        detail_dict: dict[str, Any] = {"learned": what}
        if entities:
            detail_dict["entities"] = entities
        detail_dict.update(details)
        self.stream(ActivityType.LEARNING, f"Learning: {what}", detail_dict)

    def error(self, error: str, **details: Any) -> None:
        """Stream an error activity."""
        self.stream(ActivityType.ERROR, error, details)

    def execution(self, action: str, **details: Any) -> None:
        """Stream an execution activity."""
        self.stream(ActivityType.EXECUTION, action, details)

    def context(self, message: str, **details: Any) -> None:
        """Stream a context activity."""
        self.stream(ActivityType.CONTEXT, message, details)

    def decision(self, decision: str, reasoning: str, **details: Any) -> None:
        """Stream a decision activity."""
        detail_dict: dict[str, Any] = {"decision": decision, "reasoning": reasoning}
        detail_dict.update(details)
        self.stream(ActivityType.DECISION, f"Decided: {decision}", detail_dict)


# Global activity stream instance
activity_stream = ActivityStream()
