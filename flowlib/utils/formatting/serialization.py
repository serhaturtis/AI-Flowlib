"""Object serialization utilities.

This module provides utilities for serializing complex objects
for web display or API responses, ensuring they are JSON-compatible.
"""

from typing import Any


def make_serializable(obj: Any) -> Any:
    """Convert complex objects to JSON serializable values.

    This recursively processes objects to ensure they can be safely
    serialized to JSON. It handles:
    - Objects with __dict__ attributes
    - Objects with to_dict() methods
    - Dictionaries, lists, and tuples
    - Basic primitive types
    - Other types are converted to strings

    Args:
        obj: Object to make serializable

    Returns:
        JSON-serializable version of the object
    """
    if obj is None:
        return None

    if hasattr(obj, "__dict__"):
        # For object with __dict__, convert to dict
        result = {}
        for k, v in obj.__dict__.items():
            if not k.startswith("_"):  # Skip private attributes
                result[k] = make_serializable(v)
        return result
    elif isinstance(obj, dict):
        # For dictionaries, recursively convert values
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # For lists/tuples, recursively convert items
        return [make_serializable(i) for i in obj]
    elif hasattr(obj, "to_dict"):
        # For objects with to_dict method
        return make_serializable(obj.to_dict())
    elif isinstance(obj, (str, int, float, bool)):
        # Basic types are already serializable
        return obj
    else:
        # For other types, convert to string
        return str(obj)


def format_execution_details(details: dict[str, Any]) -> dict[str, Any]:
    """Format execution details for web display.

    Args:
        details: Raw execution details dictionary

    Returns:
        Web-friendly serialized execution details
    """
    if not details:
        return {"error": "No execution details available"}

    state = details.get("state")

    # Basic details about progress and completion
    formatted = {
        "progress": getattr(state, "progress", 0) if state else 0,
        "complete": getattr(state, "is_complete", False) if state else False,
    }

    # Format latest plan if available
    if "latest_plan" in details:
        formatted["latest_plan"] = make_serializable(details["latest_plan"])

    # Format latest execution if available
    if "latest_execution" in details:
        formatted["latest_execution"] = make_serializable(details["latest_execution"])

    # Format latest reflection if available
    if "latest_reflection" in details:
        formatted["latest_reflection"] = make_serializable(details["latest_reflection"])

    # Format execution history
    execution_history = details["execution_history"] if "execution_history" in details else []
    formatted["execution_history"] = make_serializable(execution_history)

    # Include any additional fields that might be useful
    if "flows" in details:
        formatted["flows"] = make_serializable(details["flows"])

    return formatted
