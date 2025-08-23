"""Tests for agent activity formatter."""

import pytest
from datetime import datetime
from typing import Dict, Any

from flowlib.agent.core.agent_activity_formatter import AgentActivityFormatter


class MockEntry:
    """Mock execution history entry for testing."""
    
    def __init__(self, flow_name: str, inputs: Dict[str, Any], result: Dict[str, Any]):
        self.flow_name = flow_name
        self.inputs = inputs
        self.result = result


class TestAgentActivityFormatter:
    """Test AgentActivityFormatter functionality."""
    
    def test_format_basic_agent_activity(self):
        """Test formatting basic agent activity."""
        result = {
            "task_id": "task_123",
            "cycles": 3,
            "progress": 75,
            "is_complete": True,
            "output": "Task completed successfully"
        }
        
        formatted = AgentActivityFormatter.format_agent_activity(result)
        
        assert "🤖 Agent Activity:" in formatted
        assert "Task ID: task_123" in formatted
        assert "Cycles executed: 3" in formatted
        assert "Progress: 75%" in formatted
        assert "✅ Task completed successfully" in formatted
        assert "💬 Response: Task completed successfully" in formatted
        # Check that the output appears in both status and response sections
        assert formatted.count("Task completed successfully") == 2
    
    def test_format_agent_activity_with_output(self):
        """Test formatting agent activity with custom output."""
        result = {
            "task_id": "task_456",
            "cycles": 2,
            "progress": 50,
            "is_complete": False,
            "output": "Analysis complete: Found 5 issues to resolve."
        }
        
        formatted = AgentActivityFormatter.format_agent_activity(result)
        
        assert "Task ID: task_456" in formatted
        assert "⏳ Task in progress..." in formatted
        assert "💬 Response: Analysis complete: Found 5 issues to resolve." in formatted
    
    def test_format_agent_activity_with_execution_history_dict(self):
        """Test formatting with execution history as dictionaries."""
        result = {
            "task_id": "task_789",
            "cycles": 1,
            "progress": 100,
            "execution_history": [
                {
                    "flow_name": "shell-command",
                    "inputs": {"message": "ls -la"},
                    "result": {"status": "SUCCESS", "data": {"output": "file list"}}
                },
                {
                    "flow_name": "conversation",
                    "inputs": {"user_input": "What files are there?"},
                    "result": {"status": "SUCCESS", "data": {"response": "I found several files in the directory."}}
                }
            ],
            "is_complete": True
        }
        
        formatted = AgentActivityFormatter.format_agent_activity(result)
        
        assert "🎯 Execution Steps:" in formatted
        assert "1. shell-command" in formatted
        assert "Input: \"ls -la\"" in formatted
        assert "Status: SUCCESS" in formatted
        assert "2. conversation" in formatted
        assert "Response: \"I found several files in the directory.\"" in formatted
    
    def test_format_agent_activity_with_execution_history_objects(self):
        """Test formatting with execution history as objects."""
        result = {
            "task_id": "task_obj",
            "cycles": 2,
            "progress": 80,
            "execution_history": [
                MockEntry(
                    "task-planning",
                    {"task_description": "Plan the project structure"},
                    {"status": "SUCCESS", "data": {"plan": "3-step plan"}}
                ),
                MockEntry(
                    "conversation",
                    {"user_input": "Explain the plan"},
                    {"status": "SUCCESS", "data": {"response": "The plan consists of three phases."}}
                )
            ],
            "is_complete": False
        }
        
        formatted = AgentActivityFormatter.format_agent_activity(result)
        
        assert "1. task-planning" in formatted
        assert "Task: \"Plan the project structure\"" in formatted
        assert "2. conversation" in formatted
        assert "Response: \"The plan consists of three phases.\"" in formatted
    
    def test_format_agent_activity_with_conversation_response_object(self):
        """Test formatting conversation with response as object attribute."""
        class MockResponseData:
            def __init__(self, response: str):
                self.response = response
        
        result = {
            "task_id": "conv_test",
            "cycles": 1,
            "progress": 100,
            "execution_history": [
                {
                    "flow_name": "conversation",
                    "inputs": {"message": "Hello"},
                    "result": {"status": "SUCCESS", "data": MockResponseData("Hello! How can I help you?")}
                }
            ],
            "is_complete": True
        }
        
        formatted = AgentActivityFormatter.format_agent_activity(result)
        
        assert "1. conversation" in formatted
        assert "Response: \"Hello! How can I help you?\"" in formatted
    
    def test_format_agent_activity_with_conversation_no_response(self):
        """Test formatting conversation with no response found."""
        result = {
            "task_id": "conv_no_response",
            "cycles": 1,
            "progress": 100,
            "execution_history": [
                {
                    "flow_name": "conversation",
                    "inputs": {"message": "Hello"},
                    "result": {"status": "SUCCESS", "data": {"other_field": "value"}}
                }
            ],
            "is_complete": True
        }
        
        formatted = AgentActivityFormatter.format_agent_activity(result)
        
        assert "1. conversation" in formatted
        assert "Response: (no response found)" in formatted
    
    def test_format_agent_activity_with_errors(self):
        """Test formatting agent activity with errors."""
        result = {
            "task_id": "error_task",
            "cycles": 2,
            "progress": 30,
            "errors": [
                "Failed to connect to database",
                "Timeout occurred during API call"
            ],
            "is_complete": False
        }
        
        formatted = AgentActivityFormatter.format_agent_activity(result)
        
        assert "❌ Errors encountered:" in formatted
        assert "Failed to connect to database" in formatted
        assert "Timeout occurred during API call" in formatted
    
    def test_format_agent_activity_minimal_data(self):
        """Test formatting with minimal data."""
        result = {}
        
        formatted = AgentActivityFormatter.format_agent_activity(result)
        
        assert "Task ID: unknown" in formatted
        assert "Cycles executed: 0" in formatted
        assert "Progress: 0%" in formatted
        assert "⏳ Task in progress..." in formatted
    
    def test_format_agent_activity_empty_execution_history(self):
        """Test formatting with empty execution history."""
        result = {
            "task_id": "empty_history",
            "cycles": 0,
            "progress": 0,
            "execution_history": [],
            "is_complete": False
        }
        
        formatted = AgentActivityFormatter.format_agent_activity(result)
        
        assert "🎯 Execution Steps:" not in formatted
        assert "Task ID: empty_history" in formatted
    
    def test_format_agent_activity_various_flow_types(self):
        """Test formatting with various flow types and input patterns."""
        result = {
            "task_id": "multi_flow",
            "cycles": 4,
            "progress": 100,
            "execution_history": [
                {
                    "flow_name": "unknown-flow",
                    "inputs": {"custom_field": "value"},
                    "result": {"status": "SUCCESS"}
                },
                {
                    "flow_name": "analysis",
                    "inputs": {"message": "Analyze this data"},
                    "result": {"status": "COMPLETED"}
                },
                {
                    "flow_name": "file-processor",
                    "inputs": {"task_description": "Process uploaded files"},
                    "result": {"status": "SUCCESS"}
                }
            ],
            "is_complete": True
        }
        
        formatted = AgentActivityFormatter.format_agent_activity(result)
        
        assert "1. unknown-flow" in formatted
        assert "Status: SUCCESS" in formatted
        assert "2. analysis" in formatted
        assert "Input: \"Analyze this data\"" in formatted
        assert "Status: COMPLETED" in formatted
        assert "3. file-processor" in formatted
        assert "Task: \"Process uploaded files\"" in formatted
    
    def test_format_planning_activity_basic(self):
        """Test formatting basic planning activity."""
        planning_info = {
            "selected_flow": "conversation",
            "reasoning": "User is asking a question that requires a conversational response."
        }
        
        formatted = AgentActivityFormatter.format_planning_activity(planning_info)
        
        assert "🧠 Planning:" in formatted
        assert "Selected: conversation" in formatted
        assert "Reasoning: User is asking a question that requires a conversational response...." in formatted
    
    def test_format_planning_activity_long_reasoning(self):
        """Test formatting planning activity with long reasoning (truncated)."""
        long_reasoning = "This is a very long reasoning text that exceeds 100 characters and should be truncated to show only the first 100 characters followed by ellipsis."
        
        planning_info = {
            "selected_flow": "analysis-flow",
            "reasoning": long_reasoning
        }
        
        formatted = AgentActivityFormatter.format_planning_activity(planning_info)
        
        assert "Selected: analysis-flow" in formatted
        assert "Reasoning: This is a very long reasoning text that exceeds 100 characters and should be truncated to show only ..." in formatted
        assert len(formatted.split("Reasoning: ")[1].split("\n")[0]) <= 104  # 100 chars + "..."
    
    def test_format_planning_activity_no_reasoning(self):
        """Test formatting planning activity without reasoning."""
        planning_info = {
            "selected_flow": "shell-command",
            "reasoning": ""
        }
        
        formatted = AgentActivityFormatter.format_planning_activity(planning_info)
        
        assert "Selected: shell-command" in formatted
        assert "Reasoning:" not in formatted
    
    def test_format_planning_activity_minimal_data(self):
        """Test formatting planning activity with minimal data."""
        planning_info = {}
        
        formatted = AgentActivityFormatter.format_planning_activity(planning_info)
        
        assert "🧠 Planning:" in formatted
        assert "Selected: none" in formatted
        assert "Reasoning:" not in formatted
    
    def test_format_planning_activity_none_reasoning(self):
        """Test formatting planning activity with None reasoning."""
        planning_info = {
            "selected_flow": "test-flow",
            "reasoning": None
        }
        
        formatted = AgentActivityFormatter.format_planning_activity(planning_info)
        
        assert "Selected: test-flow" in formatted
        assert "Reasoning:" not in formatted


class TestAgentActivityFormatterEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_format_agent_activity_none_input(self):
        """Test formatting with None input."""
        # Should handle None gracefully
        formatted = AgentActivityFormatter.format_agent_activity({})
        assert "🤖 Agent Activity:" in formatted
    
    def test_format_agent_activity_malformed_history_entry(self):
        """Test formatting with malformed execution history entry."""
        result = {
            "task_id": "malformed_test",
            "execution_history": [
                {
                    # Missing flow_name
                    "inputs": {"message": "test"},
                    "result": {"status": "SUCCESS"}
                },
                {
                    "flow_name": "valid-flow",
                    # Missing inputs and result
                }
            ]
        }
        
        formatted = AgentActivityFormatter.format_agent_activity(result)
        
        # Should handle malformed entries gracefully
        assert "1. unknown" in formatted
        assert "2. valid-flow" in formatted
    
    def test_format_agent_activity_mixed_history_types(self):
        """Test formatting with mixed history entry types."""
        result = {
            "task_id": "mixed_test",
            "execution_history": [
                # Dict entry
                {
                    "flow_name": "dict-flow",
                    "inputs": {"message": "dict input"},
                    "result": {"status": "SUCCESS"}
                },
                # Object entry
                MockEntry(
                    "object-flow",
                    {"message": "object input"},
                    {"status": "COMPLETED"}
                )
            ]
        }
        
        formatted = AgentActivityFormatter.format_agent_activity(result)
        
        assert "1. dict-flow" in formatted
        assert "Input: \"dict input\"" in formatted
        assert "2. object-flow" in formatted
        assert "Input: \"object input\"" in formatted
    
    def test_static_method_access(self):
        """Test that methods can be called as static methods."""
        # Should be able to call without instantiation
        result = {"task_id": "static_test"}
        formatted = AgentActivityFormatter.format_agent_activity(result)
        assert "Task ID: static_test" in formatted
        
        planning_info = {"selected_flow": "test"}
        planning_formatted = AgentActivityFormatter.format_planning_activity(planning_info)
        assert "Selected: test" in planning_formatted