"""Tests for ActivityStream.

This module tests the ActivityStream class which manages real-time activity
streaming for agent operations with logging, formatting, and buffering capabilities.
"""

import pytest
import json
from unittest.mock import Mock, patch
from datetime import datetime
from typing import List, Dict, Any

from flowlib.agent.core.activity_stream import ActivityStream, ActivityType


class MockOutputHandler:
    """Mock output handler to capture activity output."""
    
    def __init__(self):
        self.outputs: List[str] = []
        self.call_count = 0
    
    def __call__(self, message: str):
        """Mock output handler implementation."""
        self.outputs.append(message)
        self.call_count += 1
    
    def clear(self):
        """Clear captured outputs."""
        self.outputs.clear()
        self.call_count = 0
    
    def get_last_output(self) -> str:
        """Get the last output message."""
        return self.outputs[-1] if self.outputs else ""
    
    def get_all_outputs(self) -> List[str]:
        """Get all output messages."""
        return self.outputs.copy()


class TestActivityStreamInitialization:
    """Test ActivityStream initialization."""

    def test_init_default_handler(self):
        """Test initialization with default print handler."""
        stream = ActivityStream()
        
        assert stream.output_handler is print
        assert stream.enabled is True
        assert stream.verbose is True
        assert stream.activity_buffer == []
        assert stream.current_indent == 0

    def test_init_custom_handler(self):
        """Test initialization with custom output handler."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        assert stream.output_handler is mock_handler
        assert stream.enabled is True
        assert stream.verbose is True

    def test_init_none_handler_defaults_to_print(self):
        """Test initialization with None handler defaults to print."""
        stream = ActivityStream(output_handler=None)
        
        assert stream.output_handler is print


class TestActivityStreamConfiguration:
    """Test ActivityStream configuration methods."""

    def test_set_output_handler(self):
        """Test setting output handler."""
        stream = ActivityStream()
        mock_handler = MockOutputHandler()
        
        stream.set_output_handler(mock_handler)
        
        assert stream.output_handler is mock_handler

    def test_enable_disable(self):
        """Test enabling and disabling stream."""
        stream = ActivityStream()
        
        assert stream.enabled is True
        
        stream.disable()
        assert stream.enabled is False
        
        stream.enable()
        assert stream.enabled is True

    def test_set_verbose(self):
        """Test setting verbose mode."""
        stream = ActivityStream()
        
        assert stream.verbose is True
        
        stream.set_verbose(False)
        assert stream.verbose is False
        
        stream.set_verbose(True)
        assert stream.verbose is True


class TestActivityStreamFormatting:
    """Test ActivityStream message formatting."""

    def test_format_activity_basic(self):
        """Test basic activity formatting."""
        stream = ActivityStream()
        
        result = stream._format_activity(ActivityType.PLANNING, "Test message")
        
        # Check that the basic components are present (don't check exact timestamp)
        assert "🧠 Planning: Test message" in result
        assert "] 🧠 Planning: Test message" in result  # Includes timestamp bracket

    def test_format_activity_with_details_verbose(self):
        """Test activity formatting with details in verbose mode."""
        stream = ActivityStream()
        stream.verbose = True
        details = {"key1": "value1", "key2": 42}
        
        with patch('flowlib.agent.core.activity_stream.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "12:34:56.789"
            
            result = stream._format_activity(ActivityType.EXECUTION, "Test message", details)
            
            assert "⚡ Execution: Test message" in result
            assert "→ key1: value1" in result
            assert "→ key2: 42" in result

    def test_format_activity_with_details_non_verbose(self):
        """Test activity formatting with details in non-verbose mode."""
        stream = ActivityStream()
        stream.verbose = False
        details = {"key1": "value1", "key2": 42}
        
        with patch('flowlib.agent.core.activity_stream.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "12:34:56.789"
            
            result = stream._format_activity(ActivityType.EXECUTION, "Test message", details)
            
            assert "⚡ Execution: Test message" in result
            assert "→ key1: value1" not in result
            assert "→ key2: 42" not in result

    def test_format_activity_with_dict_details(self):
        """Test activity formatting with dictionary details."""
        stream = ActivityStream()
        details = {"config": {"setting1": "value1", "setting2": "value2"}}
        
        with patch('flowlib.agent.core.activity_stream.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "12:34:56.789"
            
            result = stream._format_activity(ActivityType.CONTEXT, "Test message", details)
            
            assert "🌐 Context: Test message" in result
            assert "→ config:" in result
            assert '"setting1": "value1"' in result

    def test_format_activity_with_large_list(self):
        """Test activity formatting with large list gets summarized."""
        stream = ActivityStream()
        large_list = [f"item{i}" for i in range(10)]
        details = {"items": large_list}
        
        with patch('flowlib.agent.core.activity_stream.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "12:34:56.789"
            
            result = stream._format_activity(ActivityType.MEMORY_RETRIEVAL, "Test message", details)
            
            assert "[10 items]" in result

    def test_format_activity_with_indentation(self):
        """Test activity formatting respects indentation."""
        stream = ActivityStream()
        stream.current_indent = 2
        
        result = stream._format_activity(ActivityType.PLANNING, "Test message")
        
        # Check that the result starts with proper indentation (4 spaces for indent level 2)
        assert result.startswith("    [")  # 2 * "  " + timestamp start


class TestActivityStreamCore:
    """Test core ActivityStream functionality."""

    def test_stream_enabled(self):
        """Test streaming when enabled."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        stream.stream(ActivityType.PLANNING, "Test message")
        
        assert mock_handler.call_count == 1
        assert "🧠 Planning: Test message" in mock_handler.get_last_output()
        assert len(stream.activity_buffer) == 1

    def test_stream_disabled(self):
        """Test streaming when disabled."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        stream.disable()
        
        stream.stream(ActivityType.PLANNING, "Test message")
        
        assert mock_handler.call_count == 0
        assert len(stream.activity_buffer) == 0

    def test_stream_buffers_activity(self):
        """Test that activities are buffered."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        stream.stream(ActivityType.EXECUTION, "Test action", {"param": "value"})
        
        assert len(stream.activity_buffer) == 1
        buffer_item = stream.activity_buffer[0]
        assert buffer_item['type'] == ActivityType.EXECUTION
        assert buffer_item['message'] == "Test action"
        assert buffer_item['details'] == {"param": "value"}
        assert isinstance(buffer_item['timestamp'], datetime)

    def test_multiple_streams_accumulate_buffer(self):
        """Test that multiple streams accumulate in buffer."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        stream.stream(ActivityType.PLANNING, "Message 1")
        stream.stream(ActivityType.EXECUTION, "Message 2")
        stream.stream(ActivityType.ERROR, "Message 3")
        
        assert len(stream.activity_buffer) == 3
        assert mock_handler.call_count == 3


class TestActivityStreamSections:
    """Test ActivityStream section management."""

    def test_start_section(self):
        """Test starting a new section."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        stream.start_section("Test Section")
        
        outputs = mock_handler.get_all_outputs()
        assert len(outputs) == 3
        assert "═" * 50 in outputs[0]
        assert "▶ Test Section" in outputs[1]
        assert "─" * 50 in outputs[2]
        assert stream.current_indent == 1

    def test_start_section_disabled(self):
        """Test starting section when disabled."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        stream.disable()
        
        stream.start_section("Test Section")
        
        assert mock_handler.call_count == 0
        assert stream.current_indent == 0

    def test_end_section(self):
        """Test ending a section."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        stream.start_section("Test Section")
        mock_handler.clear()
        
        stream.end_section()
        
        assert mock_handler.call_count == 1
        assert "─" * 50 in mock_handler.get_last_output()
        assert stream.current_indent == 0

    def test_end_section_disabled(self):
        """Test ending section when disabled."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        stream.start_section("Test Section")
        stream.disable()
        
        stream.end_section()
        
        # Only the start section outputs should be there
        assert mock_handler.call_count == 3

    def test_end_section_no_active_section(self):
        """Test ending section when no active section."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        stream.end_section()
        
        assert mock_handler.call_count == 0
        assert stream.current_indent == 0

    def test_nested_sections(self):
        """Test nested sections with proper indentation."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        stream.start_section("Outer Section")
        assert stream.current_indent == 1
        
        stream.start_section("Inner Section")
        assert stream.current_indent == 2
        
        stream.end_section()
        assert stream.current_indent == 1
        
        stream.end_section()
        assert stream.current_indent == 0


class TestActivityStreamConvenienceMethods:
    """Test ActivityStream convenience methods."""

    def test_planning(self):
        """Test planning convenience method."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        stream.planning("Creating execution plan", steps=3, complexity="medium")
        
        output = mock_handler.get_last_output()
        assert "🧠 Planning: Creating execution plan" in output
        assert "→ steps: 3" in output
        assert "→ complexity: medium" in output

    def test_memory_retrieval(self):
        """Test memory retrieval convenience method."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        results = ["result1", "result2", "result3"]
        stream.memory_retrieval("search query", results=results, context="test")
        
        output = mock_handler.get_last_output()
        assert "🔍 Memory: Retrieving: search query" in output
        assert "→ found: 3" in output
        assert "→ samples:" in output
        assert "→ context: test" in output

    def test_memory_retrieval_no_results(self):
        """Test memory retrieval with no results."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        stream.memory_retrieval("search query")
        
        output = mock_handler.get_last_output()
        assert "🔍 Memory: Retrieving: search query" in output
        assert "→ found:" not in output

    def test_memory_store(self):
        """Test memory store convenience method."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        stream.memory_store("user_preferences", {"theme": "dark"}, namespace="config")
        
        output = mock_handler.get_last_output()
        assert "💾 Memory Store: Storing: user_preferences" in output
        assert "→ key: user_preferences" in output
        assert "→ type: dict" in output
        assert "→ namespace: config" in output

    def test_flow_selection(self):
        """Test flow selection convenience method."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        stream.flow_selection("ConversationFlow", "Best for user interaction", ["TaskFlow", "AnalysisFlow"])
        
        output = mock_handler.get_last_output()
        assert "🎯 Flow Selection: Selected flow: ConversationFlow" in output
        assert "→ selected: ConversationFlow" in output
        assert "→ reasoning: Best for user interaction" in output
        assert "→ alternatives: ['TaskFlow', 'AnalysisFlow']" in output

    def test_flow_selection_long_reasoning(self):
        """Test flow selection with long reasoning gets truncated."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        long_reasoning = "This is a very long reasoning text that should be truncated because it exceeds the maximum length limit for display purposes"
        stream.flow_selection("ConversationFlow", long_reasoning)
        
        output = mock_handler.get_last_output()
        assert "..." in output  # Should be truncated

    def test_prompt_selection(self):
        """Test prompt selection convenience method."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        variables = {"user_name": "Alice", "task": "analysis"}
        stream.prompt_selection("analysis_prompt", variables=variables)
        
        output = mock_handler.get_last_output()
        assert "📝 Prompt: Using prompt: analysis_prompt" in output
        assert "→ prompt: analysis_prompt" in output
        assert "→ variables: ['user_name', 'task']" in output

    def test_llm_call(self):
        """Test LLM call convenience method."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        prompt = "This is a test prompt for the LLM call"
        stream.llm_call("gpt-4", prompt, temperature=0.7, max_tokens=1000)
        
        output = mock_handler.get_last_output()
        assert "🤖 LLM: Calling gpt-4" in output
        assert "→ model: gpt-4" in output
        assert "→ preview: This is a test prompt for the LLM call" in output
        assert "→ temperature: 0.7" in output
        assert "→ max_tokens: 1000" in output

    def test_llm_call_long_prompt(self):
        """Test LLM call with long prompt gets truncated."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        long_prompt = "A" * 200  # Long prompt that should be truncated
        stream.llm_call("gpt-4", long_prompt)
        
        output = mock_handler.get_last_output()
        assert "..." in output  # Should be truncated

    def test_reflection(self):
        """Test reflection convenience method."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        stream.reflection("Task completed successfully", progress=85, insights=["good", "fast"])
        
        output = mock_handler.get_last_output()
        assert "🤔 Reflection: Reflecting on execution" in output
        assert "→ reflection: Task completed successfully" in output
        assert "→ progress: 85%" in output
        assert "→ insights: ['good', 'fast']" in output

    def test_todo_create(self):
        """Test TODO creation convenience method."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        stream.todo_create("Implement user authentication", priority="HIGH", estimate="4h")
        
        output = mock_handler.get_last_output()
        assert "📋 TODO Create: Creating TODO: Implement user authentication" in output
        assert "→ content: Implement user authentication" in output
        assert "→ priority: HIGH" in output
        assert "→ estimate: 4h" in output

    def test_todo_create_long_content(self):
        """Test TODO creation with long content gets truncated."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        long_content = "A" * 100  # Long content that should be truncated
        stream.todo_create(long_content)
        
        output = mock_handler.get_last_output()
        assert "..." in output  # Should be truncated

    def test_todo_update(self):
        """Test TODO update convenience method."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        stream.todo_update("todo-12345678", "COMPLETED", result="success")
        
        output = mock_handler.get_last_output()
        assert "✅ TODO Update: TODO todo-123 → COMPLETED" in output
        assert "→ id: todo-12345678" in output
        assert "→ status: COMPLETED" in output
        assert "→ result: success" in output

    def test_todo_status(self):
        """Test TODO status convenience method."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        stream.todo_status(total=10, completed=7, in_progress=2)
        
        output = mock_handler.get_last_output()
        assert "📊 TODO Status: TODOs: 7/10 completed, 2 in progress" in output

    def test_learning(self):
        """Test learning convenience method."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        entities = ["user", "project", "task"]
        stream.learning("User prefers dark mode", entities=entities, confidence=0.9)
        
        output = mock_handler.get_last_output()
        assert "🎓 Learning: Learning: User prefers dark mode" in output
        assert "→ learned: User prefers dark mode" in output
        assert "→ entities: ['user', 'project', 'task']" in output
        assert "→ confidence: 0.9" in output

    def test_error(self):
        """Test error convenience method."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        stream.error("Connection timeout", code=500, retry_count=3)
        
        output = mock_handler.get_last_output()
        assert "❌ Error: Connection timeout" in output
        assert "→ code: 500" in output
        assert "→ retry_count: 3" in output

    def test_execution(self):
        """Test execution convenience method."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        stream.execution("Running data analysis", tool="pandas", duration="2.5s")
        
        output = mock_handler.get_last_output()
        assert "⚡ Execution: Running data analysis" in output
        assert "→ tool: pandas" in output
        assert "→ duration: 2.5s" in output

    def test_context(self):
        """Test context convenience method."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        stream.context("Switching to user context", user_id="123", session="abc")
        
        output = mock_handler.get_last_output()
        assert "🌐 Context: Switching to user context" in output
        assert "→ user_id: 123" in output
        assert "→ session: abc" in output

    def test_decision(self):
        """Test decision convenience method."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        stream.decision("Use caching", "Improves performance", impact="high")
        
        output = mock_handler.get_last_output()
        assert "💭 Decision: Decided: Use caching" in output
        assert "→ decision: Use caching" in output
        assert "→ reasoning: Improves performance" in output
        assert "→ impact: high" in output


class TestActivityStreamIntegration:
    """Integration tests for ActivityStream with complex scenarios."""

    def test_complete_workflow_simulation(self):
        """Test a complete workflow simulation with multiple activity types."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        # Simulate a complete agent workflow
        stream.start_section("User Request Processing")
        stream.planning("Analyzing user request", complexity="medium")
        stream.memory_retrieval("similar requests", results=["req1", "req2"])
        stream.flow_selection("TaskFlow", "Best for complex tasks")
        stream.execution("Processing request")
        stream.reflection("Task completed", progress=100)
        stream.end_section()
        
        outputs = mock_handler.get_all_outputs()
        
        # Should have section markers + activities
        assert len(outputs) > 6
        assert any("▶ User Request Processing" in output for output in outputs)
        assert any("🧠 Planning" in output for output in outputs)
        assert any("🔍 Memory" in output for output in outputs)
        assert any("🎯 Flow Selection" in output for output in outputs)
        assert any("⚡ Execution" in output for output in outputs)
        assert any("🤔 Reflection" in output for output in outputs)

    def test_verbose_vs_non_verbose_output_difference(self):
        """Test difference between verbose and non-verbose output."""
        mock_handler_verbose = MockOutputHandler()
        mock_handler_non_verbose = MockOutputHandler()
        
        stream_verbose = ActivityStream(output_handler=mock_handler_verbose)
        stream_verbose.set_verbose(True)
        
        stream_non_verbose = ActivityStream(output_handler=mock_handler_non_verbose)
        stream_non_verbose.set_verbose(False)
        
        details = {"key1": "value1", "key2": "value2"}
        
        stream_verbose.execution("Test action", **details)
        stream_non_verbose.execution("Test action", **details)
        
        verbose_output = mock_handler_verbose.get_last_output()
        non_verbose_output = mock_handler_non_verbose.get_last_output()
        
        # Verbose should include details, non-verbose should not
        assert "→ key1: value1" in verbose_output
        assert "→ key1: value1" not in non_verbose_output
        # Both should contain the main message
        assert "Test action" in verbose_output
        assert "Test action" in non_verbose_output

    def test_nested_sections_with_activities(self):
        """Test nested sections with activities and proper indentation."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        stream.start_section("Main Task")
        stream.planning("Main plan")
        
        stream.start_section("Subtask 1")
        stream.execution("Sub action 1")
        stream.end_section()
        
        stream.start_section("Subtask 2")
        stream.execution("Sub action 2")
        stream.end_section()
        
        stream.reflection("Main task complete")
        stream.end_section()
        
        outputs = mock_handler.get_all_outputs()
        
        # Find planning and execution activities
        planning_output = next(output for output in outputs if "🧠 Planning" in output)
        sub_execution = next(output for output in outputs if "Sub action 1" in output)
        main_reflection = next(output for output in outputs if "Main task complete" in output)
        
        # Check indentation levels
        assert planning_output.startswith("  [")  # 1 level indent
        assert sub_execution.startswith("    [")  # 2 level indent
        assert main_reflection.startswith("  [")  # Back to 1 level indent

    def test_activity_buffer_accumulation(self):
        """Test that activity buffer properly accumulates all activities."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        
        # Add various activities
        stream.planning("Plan A")
        stream.execution("Execute B")
        stream.error("Error C")
        stream.reflection("Reflect D")
        
        assert len(stream.activity_buffer) == 4
        
        # Check buffer contents
        buffer_messages = [item['message'] for item in stream.activity_buffer]
        assert "Plan A" in buffer_messages
        assert "Execute B" in buffer_messages
        assert "Error C" in buffer_messages
        assert "Reflecting on execution" in buffer_messages  # reflection() uses this message
        
        # Check buffer types
        buffer_types = [item['type'] for item in stream.activity_buffer]
        assert ActivityType.PLANNING in buffer_types
        assert ActivityType.EXECUTION in buffer_types
        assert ActivityType.ERROR in buffer_types
        assert ActivityType.REFLECTION in buffer_types

    def test_disabled_stream_no_buffer_accumulation(self):
        """Test that disabled stream doesn't accumulate buffer."""
        mock_handler = MockOutputHandler()
        stream = ActivityStream(output_handler=mock_handler)
        stream.disable()
        
        stream.planning("Plan A")
        stream.execution("Execute B")
        
        assert len(stream.activity_buffer) == 0
        assert mock_handler.call_count == 0

    def test_global_activity_stream_import(self):
        """Test that global activity stream can be imported."""
        from flowlib.agent.core.activity_stream import activity_stream
        
        assert isinstance(activity_stream, ActivityStream)
        assert activity_stream.output_handler is print


if __name__ == "__main__":
    pytest.main([__file__, "-v"])