"""Conversation formatting utilities.

This module provides utilities for formatting conversation data,
including message history, state information, and flow representations.
"""

from typing import Any


def format_conversation(conversation: list[dict[str, str]]) -> str:
    """Format conversation history into a string for prompts.

    Args:
        conversation: List of message dictionaries with 'speaker' and 'content' keys

    Returns:
        Formatted conversation string
    """
    if not conversation:
        return ""

    formatted = []
    for message in conversation:
        speaker = message["speaker"] if "speaker" in message else "Unknown"
        content = message["content"] if "content" in message else ""
        formatted.append(f"{speaker}: {content}")

    return "\n".join(formatted)


def format_state(state: dict[str, Any]) -> str:
    """Format agent state for prompt.

    Args:
        state: Dictionary of state variables

    Returns:
        Formatted state string
    """
    if not state:
        return "No state information."

    result = []
    for key, value in state.items():
        result.append(f"{key}: {value}")

    return "\n".join(result)


def format_history(history: list[dict[str, Any]]) -> str:
    """Format execution history for prompt.

    Args:
        history: List of execution steps

    Returns:
        Formatted history string
    """
    # Normalize history format to match the expectations of format_execution_history
    if not history:
        return "No execution history yet."

    # Convert from the old format to the new standardized format
    normalized_history: list[dict[str, Any]] = []
    for step in history:
        if "action" in step and step["action"] == "execute_flow":
            normalized_entry = {
                "flow_name": step["flow"] if "flow" in step else "unknown",
                "cycle": len(normalized_history) + 1,
                "status": "completed",
                "reasoning": step["reasoning"] if "reasoning" in step else "",
                "reflection": step["reflection"] if "reflection" in step else "",
            }
            normalized_history.append(normalized_entry)
        elif "action" in step and step["action"] == "error":
            normalized_entry = {
                "flow_name": "error",
                "cycle": len(normalized_history) + 1,
                "status": "error",
                "reasoning": "",
                "reflection": step["error"] if "error" in step else "Unknown error",
            }
            normalized_history.append(normalized_entry)

    # Use the standardized function
    return format_execution_history(normalized_history)


def format_flows(flows: list[dict[str, Any]]) -> str:
    """Format available flows for prompt.

    Args:
        flows: List of flow information dictionaries

    Returns:
        Formatted flows string
    """
    if not flows:
        return "No flows available."

    result = ["Available flows:"]

    for flow in sorted(flows, key=lambda f: f["name"] if "name" in f else ""):
        name = flow["name"] if "name" in flow else "Unknown"
        description = flow["description"] if "description" in flow else "No description available."

        # Add schema info if available
        schema_str = ""
        if "schema" in flow:
            schema = flow["schema"]
            if "input" in schema:
                input_schema = schema["input"]
                schema_str = f"\n  Input: {input_schema}"

            if "output" in schema:
                output_schema = schema["output"]
                schema_str += f"\n  Output: {output_schema}"

        result.append(f"\n{name}: {description}{schema_str}")

    # Add a note about flow names
    result.append("\nNOTE: When selecting a flow, use the exact flow name without any quotes.")

    return "\n".join(result)


def format_execution_history(history_entries: list[dict[str, Any]]) -> str:
    """Format execution history from agent state or history entries into a standardized text format.

    This is the standard function for formatting execution history for prompts
    across all agent components.

    Args:
        history_entries: List of execution history entries, typically from AgentState.execution_history

    Returns:
        Formatted history string
    """
    if not history_entries or len(history_entries) == 0:
        return "No execution history available"

    history_items = []
    for i, entry in enumerate(history_entries, 1):
        if isinstance(entry, dict):
            # Extract standard fields with strict validation
            flow = entry["flow_name"] if "flow_name" in entry else "unknown"
            cycle = entry["cycle"] if "cycle" in entry else 0

            # Extract status - handling both formats that might be used
            status = "unknown"
            if "status" in entry:
                status = entry["status"]
            elif "result" in entry and isinstance(entry["result"], dict):
                status = entry["result"]["status"] if "status" in entry["result"] else "unknown"

            # Get reasoning if available
            reasoning = ""
            if "reasoning" in entry:
                reasoning = entry["reasoning"]
            elif "inputs" in entry and isinstance(entry["inputs"], dict):
                reasoning = entry["inputs"]["reasoning"] if "reasoning" in entry["inputs"] else ""

            # Get reflection if available
            reflection = ""
            if "reflection" in entry:
                reflection = entry["reflection"]
            elif "result" in entry and isinstance(entry["result"], dict):
                reflection = (
                    entry["result"]["reflection"] if "reflection" in entry["result"] else ""
                )

            # Format using all available information
            entry_text = f"{i}. Cycle {cycle}: Executed {flow} with status {status}"

            # Add reasoning and reflection if available
            if reasoning:
                entry_text += f"\n   Reasoning: {reasoning}"
            if reflection:
                entry_text += f"\n   Reflection: {reflection}"

            history_items.append(entry_text)

    return "\n".join(history_items)


def format_agent_execution_details(details: dict[str, Any]) -> str:
    """Format agent execution details for CLI display.

    Args:
        details: Dictionary of execution details

    Returns:
        Formatted execution details string
    """
    if not details:
        return "No detailed agent execution information available."

    result = ["--- Agent Execution Details ---"]

    if "state" in details:
        state = details["state"]
        # Process state information
        # Get progress value - normalize to percentage if needed
        progress = getattr(state, "progress", 0)
        # Check if progress is already a percentage (0-100) or a fraction (0-1)
        if progress <= 1.0:  # If it's a fraction, convert to percentage
            progress_display = f"{progress * 100:.0f}%"
        else:  # It's already a percentage
            progress_display = f"{progress:.0f}%"

        is_complete = getattr(state, "is_complete", False)

        result.append(f"Progress: {progress_display}")
        result.append(f"Complete: {'Yes' if is_complete else 'No'}")

    # Format latest plan
    if "latest_plan" in details:
        latest_plan = details["latest_plan"]
        result.append("\nLatest plan:")
        reasoning = latest_plan["reasoning"] if "reasoning" in latest_plan else "No reasoning"
        flow = latest_plan["flow"] if "flow" in latest_plan else "No flow selected"
        result.append(f"  Reasoning: {reasoning}")
        result.append(f"  Selected flow: {flow}")

    # Format last execution
    if "latest_execution" in details:
        execution = details["latest_execution"]
        result.append("\nLatest execution:")
        action = execution["action"] if "action" in execution else "unknown"
        flow = execution["flow"] if "flow" in execution else "unknown"
        result.append(f"  Action: {action}")
        result.append(f"  Flow: {flow}")

    # Format latest reflection
    if "latest_reflection" in details:
        latest_reflection = details["latest_reflection"]
        result.append("\nLatest reflection:")
        reflection = (
            latest_reflection["reflection"]
            if "reflection" in latest_reflection
            else "No reflection available"
        )
        result.append(f"  {reflection}")

    result.append("-----------------------------")

    return "\n".join(result)
