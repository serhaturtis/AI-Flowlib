"""
User display interface for flow outputs.

This module provides a standardized way for flows to specify how their
outputs should be displayed to end users.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class UserDisplayable(Protocol):
    """Protocol for objects that can provide user-friendly display text.

    Any flow output model that implements this protocol can control
    how its results are presented to the user.

    Note: This is a Protocol, so classes just need to implement the
    get_user_display() method - no inheritance required.
    """

    def get_user_display(self) -> str:
        """Get user-friendly display text for this result.

        Returns:
            Human-readable string describing the result
        """
        ...


def extract_user_display(result_data: object) -> str | None:
    """Extract user display text from any result object.

    This function provides a standardized way to get user-friendly
    text from any flow result, regardless of its specific type.

    Args:
        result_data: Result object from a flow execution

    Returns:
        User-friendly display text, or None if not available
    """
    # Check if it implements UserDisplayable protocol
    if isinstance(result_data, UserDisplayable):
        try:
            display_text = result_data.get_user_display()
            return str(display_text) if display_text is not None else None
        except Exception as e:
            # If method exists but fails, this is a critical error - don't mask it
            raise RuntimeError(f"get_user_display() method failed: {str(e)}") from e

    # Handle known types for display field extraction
    if isinstance(result_data, dict):
        data_dict = result_data
    else:
        # Convert unknown objects to string representation
        return str(result_data)

    # Look for common display fields in order of preference
    display_fields = [
        "user_display",
        "display_text",
        "summary",
        "message",
        "response",
        "output",
        "result",
    ]

    for field in display_fields:
        if field in data_dict and data_dict[field]:
            value = data_dict[field]
            if isinstance(value, str):
                return value

    return None


def format_flow_output_for_user(flow_name: str, result_data: object, success: bool = True) -> str:
    """Format any flow output for user display with fallback formatting.

    This is the main function used by the agent engine to format
    flow outputs in a standardized way.

    Args:
        flow_name: Name of the flow that produced the result
        result_data: Result data from the flow
        success: Whether the flow execution was successful

    Returns:
        User-friendly formatted output
    """
    # Try to get user display text from the result
    user_display = extract_user_display(result_data)
    if user_display:
        return user_display

    # Fallback to flow-specific formatting for known flow types
    if flow_name == "shell-command":
        return _format_shell_command_output(result_data, success)
    elif flow_name == "conversation":
        return _format_conversation_output(result_data, success)

    # Generic fallback formatting
    if success:
        return f"✅ {flow_name} completed successfully"
    else:
        # Extract error message from dict or object - strict access
        error_msg = "Unknown error"
        if isinstance(result_data, dict):
            if "error" in result_data:
                error_msg = str(result_data["error"])
        else:
            # Try to extract error from object with error attribute
            try:
                error_attr = getattr(result_data, "error", None)
                if error_attr is not None:
                    error_msg = str(error_attr)
            except (AttributeError, TypeError):
                pass
        return f"❌ {flow_name} failed: {error_msg}"


def _format_shell_command_output(result_data: object, success: bool) -> str:
    """Format shell command output for user display."""
    if isinstance(result_data, dict):
        data = result_data
    else:
        return "Shell command executed"

    command = data["command"] if "command" in data else None
    stdout = data["stdout"] if "stdout" in data else ""
    stderr = data["stderr"] if "stderr" in data else ""

    if not command:
        return "Shell command executed"

    if success and stdout.strip():
        return f"Command executed successfully:\n```bash\n$ {command}\n{stdout}\n```"
    elif not success and stderr.strip():
        return f"Command failed:\n```bash\n$ {command}\n\nError: {stderr}\n```"
    else:
        return f"Command executed:\n```bash\n$ {command}\n\n(No output)\n```"


def _format_conversation_output(result_data: object, success: bool) -> str:
    """Format conversation output for user display."""
    if isinstance(result_data, dict):
        data = result_data
    else:
        return "Conversation completed"

    response = ""
    if "response" in data:
        response = data["response"]
    elif "message" in data:
        response = data["message"]
    if response:
        return response

    return "Conversation completed"
