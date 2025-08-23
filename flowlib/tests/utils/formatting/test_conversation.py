"""Tests for conversation formatting utilities."""

import pytest
from datetime import datetime
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from flowlib.utils.formatting.conversation import format_conversation, format_state, format_history, format_flows, format_execution_history, format_agent_execution_details


# Test models - rename to avoid pytest confusion
class MessageModel(BaseModel):
    """Test message for conversation formatting."""
    id: str
    content: str
    sender: str
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationModel(BaseModel):
    """Test conversation model."""
    id: str
    messages: List[MessageModel]
    participants: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TestFormatConversation:
    """Test conversation formatting utilities."""
    
    def test_format_conversation_basic(self):
        """Test basic conversation formatting."""
        conversation = [
            {"speaker": "user", "content": "Hello"},
            {"speaker": "assistant", "content": "Hi there!"}
        ]
        
        formatted = format_conversation(conversation)
        assert isinstance(formatted, str)
        assert "Hello" in formatted
        assert "Hi there!" in formatted
        assert "user" in formatted
        assert "assistant" in formatted
    
    def test_format_conversation_empty(self):
        """Test formatting empty conversation."""
        formatted = format_conversation([])
        assert formatted == ""
    
    def test_format_conversation_with_metadata(self):
        """Test formatting conversation with additional fields."""
        conversation = [
            {"speaker": "user", "content": "Test message", "extra": "ignored"},
            {"speaker": "assistant", "content": "Response"}
        ]
        
        formatted = format_conversation(conversation)
        assert isinstance(formatted, str)
        assert "Test message" in formatted
        assert "Response" in formatted


class TestFormatState:
    """Test state formatting utilities."""
    
    def test_format_state_basic(self):
        """Test basic state formatting."""
        state = {
            "current_step": "processing",
            "progress": 75,
            "items_processed": 150,
            "errors": [],
            "metadata": {"started_at": "2023-01-01T12:00:00"}
        }
        
        formatted = format_state(state)
        assert isinstance(formatted, str)
        assert "current_step" in formatted
        assert "processing" in formatted
        assert "75" in formatted
    
    def test_format_state_with_errors(self):
        """Test formatting state with errors."""
        state = {
            "current_step": "failed",
            "progress": 50,
            "errors": ["Connection timeout", "Invalid response"],
            "last_error": "Connection timeout"
        }
        
        formatted = format_state(state)
        assert isinstance(formatted, str)
        assert "failed" in formatted
        assert "Connection timeout" in formatted
        assert "errors" in formatted
    
    def test_format_state_empty(self):
        """Test formatting empty state."""
        formatted = format_state({})
        assert formatted == "No state information."
    
    def test_format_state_nested(self):
        """Test formatting nested state."""
        state = {
            "process": {
                "name": "data_processing",
                "status": "running",
                "substeps": {
                    "validation": "completed",
                    "transformation": "in_progress",
                    "storage": "pending"
                }
            },
            "metrics": {
                "throughput": 1000,
                "latency": 50.5
            }
        }
        
        formatted = format_state(state)
        assert isinstance(formatted, str)
        # Note: format_state doesn't handle nested objects specially
        assert "process" in formatted
        assert "metrics" in formatted


class TestFormatHistory:
    """Test history formatting utilities."""
    
    def test_format_history_basic(self):
        """Test basic history formatting."""
        history = [
            {"action": "execute_flow", "flow": "test_flow", "reasoning": "Test reasoning"},
            {"action": "error", "error": "Test error"}
        ]
        
        formatted = format_history(history)
        assert isinstance(formatted, str)
        assert "test_flow" in formatted
        assert "error" in formatted
    
    def test_format_history_empty(self):
        """Test formatting empty history."""
        formatted = format_history([])
        assert formatted == "No execution history yet."


class TestFormatFlows:
    """Test flows formatting utilities."""
    
    def test_format_flows_basic(self):
        """Test basic flows formatting."""
        flows = [
            {
                "name": "test_flow",
                "description": "A test flow",
                "schema": {
                    "input": {"type": "string"},
                    "output": {"type": "object"}
                }
            }
        ]
        
        formatted = format_flows(flows)
        assert isinstance(formatted, str)
        assert "test_flow" in formatted
        assert "A test flow" in formatted
        assert "Input:" in formatted
        assert "Output:" in formatted
    
    def test_format_flows_empty(self):
        """Test formatting empty flows."""
        formatted = format_flows([])
        assert formatted == "No flows available."


class TestFormatExecutionHistory:
    """Test execution history formatting utilities."""
    
    def test_format_execution_history_basic(self):
        """Test basic execution history formatting."""
        history = [
            {
                "flow_name": "test_flow",
                "cycle": 1,
                "status": "completed",
                "reasoning": "Test reasoning",
                "reflection": "Test reflection"
            }
        ]
        
        formatted = format_execution_history(history)
        assert isinstance(formatted, str)
        assert "test_flow" in formatted
        assert "completed" in formatted
        assert "Test reasoning" in formatted
        assert "Test reflection" in formatted
    
    def test_format_execution_history_empty(self):
        """Test formatting empty execution history."""
        formatted = format_execution_history([])
        assert formatted == "No execution history available"


class TestFormatAgentExecutionDetails:
    """Test agent execution details formatting utilities."""
    
    def test_format_agent_execution_details_basic(self):
        """Test basic agent execution details formatting."""
        from unittest.mock import Mock
        
        state = Mock()
        state.progress = 75
        state.is_complete = False
        
        details = {
            "state": state,
            "latest_plan": {
                "reasoning": "Test reasoning",
                "flow": "test_flow"
            },
            "latest_execution": {
                "action": "execute",
                "flow": "test_flow"
            },
            "latest_reflection": {
                "reflection": "Test reflection"
            }
        }
        
        formatted = format_agent_execution_details(details)
        assert isinstance(formatted, str)
        assert "75%" in formatted
        assert "Test reasoning" in formatted
        assert "test_flow" in formatted
        assert "Test reflection" in formatted
    
    def test_format_agent_execution_details_empty(self):
        """Test formatting empty agent execution details."""
        formatted = format_agent_execution_details({})
        assert formatted == "No detailed agent execution information available."