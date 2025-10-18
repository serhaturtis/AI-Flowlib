"""Formatter for displaying agent activity in REPL."""

from typing import Any, Dict, List


class AgentActivityFormatter:
    """Formats agent execution details for REPL display."""

    @staticmethod
    def format_agent_activity(result: Dict[str, Any]) -> str:
        """Format agent execution result to show what the agent did.
        
        Args:
            result: The execution result from the agent
            
        Returns:
            Formatted string showing agent activity
        """
        lines = []

        # Header
        lines.append("\n🤖 Agent Activity:")
        lines.append("=" * 50)

        # Task info
        task_id = result["task_id"] if "task_id" in result else "unknown"
        cycles = result["cycles"] if "cycles" in result else 0
        progress = result["progress"] if "progress" in result else 0

        lines.append(f"📋 Task ID: {task_id}")
        lines.append(f"🔄 Cycles executed: {cycles}")
        lines.append(f"📊 Progress: {progress}%")

        # Execution history
        history = result["execution_history"] if "execution_history" in result else []
        if history:
            lines.append("\n🎯 Execution Steps:")
            for i, entry in enumerate(history, 1):
                # Handle both dict and object formats
                if isinstance(entry, dict):
                    flow_name = entry['flow_name'] if 'flow_name' in entry else 'unknown'
                    inputs = entry['inputs'] if 'inputs' in entry else {}
                    result_data = entry['result'] if 'result' in entry else {}
                else:
                    flow_name = entry.flow_name if hasattr(entry, 'flow_name') else "unknown"
                    inputs = entry.inputs if hasattr(entry, 'inputs') else {}
                    result_data = entry.result if hasattr(entry, 'result') else {}

                lines.append(f"  {i}. {flow_name}")

                # Show key inputs/outputs based on flow type
                if flow_name == 'conversation':
                    # For conversation flows, show the response instead of input
                    if isinstance(result_data, dict):
                        data = result_data['data'] if 'data' in result_data else {}
                        if isinstance(data, dict) and 'response' in data:
                            lines.append(f"     Response: \"{data['response']}\"")
                        elif hasattr(data, 'response'):
                            lines.append(f"     Response: \"{data.response}\"")
                        else:
                            lines.append("     Response: (no response found)")
                else:
                    # For other flows, show inputs
                    if isinstance(inputs, dict):
                        if 'message' in inputs:
                            lines.append(f"     Input: \"{inputs['message']}\"")
                        elif 'task_description' in inputs:
                            lines.append(f"     Task: \"{inputs['task_description']}\"")

                # Show result status
                if isinstance(result_data, dict):
                    status = result_data['status'] if 'status' in result_data else 'unknown'
                    lines.append(f"     Status: {status}")

        # Errors if any
        errors = result["errors"] if "errors" in result else []
        if errors:
            lines.append("\n❌ Errors encountered:")
            for error in errors:
                lines.append(f"  - {error}")

        # Completion status
        is_complete = result["is_complete"] if "is_complete" in result else False
        if is_complete:
            lines.append("\n✅ Task completed successfully")
        else:
            lines.append("\n⏳ Task in progress...")

        lines.append("=" * 50)

        # The actual response
        output = result["output"] if "output" in result else ""
        if output and output != "Task completed":
            lines.append(f"\n💬 Response: {output}")

        return "\n".join(lines)

    @staticmethod
    def format_activity_stream(activity_items: List[Dict[str, Any]]) -> str:
        """Format activity stream items for display.
        
        Args:
            activity_items: List of activity items from ActivityStream.activity_buffer
            
        Returns:
            Formatted activity stream output
        """
        if not activity_items:
            return ""

        lines = []
        lines.append("🔄 Activity Stream:")
        lines.append("─" * 40)

        for item in activity_items:
            # Extract fields safely
            timestamp = item.get('timestamp')
            activity_type = item.get('type')
            message = item.get('message', '')
            details = item.get('details', {})

            # Format timestamp
            time_str = ""
            if timestamp:
                time_str = f"[{timestamp.strftime('%H:%M:%S')}] "

            # Format activity type
            type_str = ""
            if activity_type:
                # Get the emoji and name from ActivityType enum
                type_str = f"{activity_type.value}: " if hasattr(activity_type, 'value') else f"{str(activity_type)}: "

            # Main activity line
            lines.append(f"  {time_str}{type_str}{message}")

            # Add relevant details
            if details:
                for key, value in details.items():
                    if key in ['inputs', 'result', 'query', 'selected', 'decision', 'content']:
                        # Show important details
                        value_str = str(value)[:80] + "..." if len(str(value)) > 80 else str(value)
                        lines.append(f"    → {key}: {value_str}")

        lines.append("─" * 40)
        return "\n".join(lines)

    @staticmethod
    def format_planning_activity(planning_info: Dict[str, Any]) -> str:
        """Format planning activity for display.
        
        Args:
            planning_info: Planning information
            
        Returns:
            Formatted planning activity
        """
        lines = []
        lines.append("🧠 Planning:")

        selected_flow = planning_info["selected_flow"] if "selected_flow" in planning_info else "none"
        reasoning = planning_info["reasoning"] if "reasoning" in planning_info else ""

        lines.append(f"  Selected: {selected_flow}")
        if reasoning:
            lines.append(f"  Reasoning: {reasoning[:100]}...")

        return "\n".join(lines)
