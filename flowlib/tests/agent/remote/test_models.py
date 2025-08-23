"""Tests for remote agent models."""

import pytest
from datetime import datetime
from typing import Dict, Any

from flowlib.agent.runners.remote.models import AgentTaskMessage, AgentResultMessage


class TestAgentTaskMessage:
    """Test AgentTaskMessage model."""
    
    def test_task_message_basic(self):
        """Test basic task message creation."""
        task_msg = AgentTaskMessage(
            task_id="task_123",
            task_description="Process user data"
        )
        
        assert task_msg.task_id == "task_123"
        assert task_msg.task_description == "Process user data"
        assert task_msg.initial_state_id is None
        assert task_msg.initial_state_data is None
        assert task_msg.agent_config is None
        assert task_msg.reply_to_queue is None
        assert task_msg.correlation_id is None
    
    def test_task_message_with_state(self):
        """Test task message with initial state."""
        initial_data = {
            "progress": 0.5,
            "current_step": "processing",
            "metadata": {"source": "api"}
        }
        
        task_msg = AgentTaskMessage(
            task_id="task_456",
            task_description="Continue processing",
            initial_state_id="state_789",
            initial_state_data=initial_data
        )
        
        assert task_msg.initial_state_id == "state_789"
        assert task_msg.initial_state_data == initial_data
        assert task_msg.initial_state_data["progress"] == 0.5
    
    def test_task_message_with_config_overrides(self):
        """Test task message with agent configuration overrides."""
        config_overrides = {
            "max_cycles": 10,
            "llm_provider": "openai",
            "temperature": 0.7
        }
        
        task_msg = AgentTaskMessage(
            task_id="task_config",
            task_description="Test with custom config",
            agent_config=config_overrides,
            reply_to_queue="custom_results",
            correlation_id="corr_123"
        )
        
        assert task_msg.agent_config == config_overrides
        assert task_msg.agent_config["max_cycles"] == 10
        assert task_msg.reply_to_queue == "custom_results"
        assert task_msg.correlation_id == "corr_123"
    
    def test_task_message_serialization(self):
        """Test task message JSON serialization/deserialization."""
        task_msg = AgentTaskMessage(
            task_id="serialize_test",
            task_description="Test serialization",
            initial_state_data={"key": "value"},
            agent_config={"setting": True},
            correlation_id="test_corr"
        )
        
        # Serialize to JSON
        json_str = task_msg.model_dump_json()
        assert "serialize_test" in json_str
        assert "Test serialization" in json_str
        
        # Deserialize from JSON
        restored_msg = AgentTaskMessage.model_validate_json(json_str)
        assert restored_msg.task_id == task_msg.task_id
        assert restored_msg.task_description == task_msg.task_description
        assert restored_msg.initial_state_data == task_msg.initial_state_data
        assert restored_msg.agent_config == task_msg.agent_config
        assert restored_msg.correlation_id == task_msg.correlation_id
    
    def test_task_message_extra_fields(self):
        """Test that extra fields are allowed."""
        data = {
            "task_id": "extra_test",
            "task_description": "Test extra fields",
            "custom_field": "custom_value",
            "priority": "high"
        }
        
        task_msg = AgentTaskMessage.model_validate(data)
        assert task_msg.task_id == "extra_test"
        # Extra fields should be preserved due to Config.extra = "allow"
        
    def test_task_message_validation_errors(self):
        """Test validation errors for required fields."""
        # Missing task_id - Pydantic ValidationError
        with pytest.raises(Exception):  # Pydantic raises ValidationError (subclass of ValueError)
            AgentTaskMessage(task_description="Missing task_id")
        
        # Missing task_description
        with pytest.raises(Exception):
            AgentTaskMessage(task_id="test_id")
        
        # Empty task_id should be valid (just test that it works)
        task_msg = AgentTaskMessage(task_id="", task_description="Empty task_id")
        assert task_msg.task_id == ""


class TestAgentResultMessage:
    """Test AgentResultMessage model."""
    
    def test_result_message_success(self):
        """Test successful result message."""
        result_data = {
            "output": "Task completed successfully",
            "metrics": {"execution_time": 12.5},
            "files_processed": 5
        }
        
        result_msg = AgentResultMessage(
            task_id="task_success",
            status="SUCCESS",
            final_state_id="final_state_123",
            result_data=result_data,
            correlation_id="corr_success"
        )
        
        assert result_msg.task_id == "task_success"
        assert result_msg.status == "SUCCESS"
        assert result_msg.final_state_id == "final_state_123"
        assert result_msg.result_data == result_data
        assert result_msg.error_message is None
        assert result_msg.correlation_id == "corr_success"
    
    def test_result_message_failure(self):
        """Test failure result message."""
        result_msg = AgentResultMessage(
            task_id="task_failure",
            status="FAILURE",
            error_message="Connection timeout occurred",
            correlation_id="corr_failure"
        )
        
        assert result_msg.status == "FAILURE"
        assert result_msg.error_message == "Connection timeout occurred"
        assert result_msg.final_state_id is None
        assert result_msg.result_data is None
    
    def test_result_message_incomplete(self):
        """Test incomplete result message."""
        partial_data = {
            "progress": 0.75,
            "completed_steps": 3,
            "remaining_steps": 1
        }
        
        result_msg = AgentResultMessage(
            task_id="task_incomplete",
            status="INCOMPLETE",
            final_state_id="partial_state",
            result_data=partial_data,
            error_message="Max cycles reached"
        )
        
        assert result_msg.status == "INCOMPLETE"
        assert result_msg.result_data["progress"] == 0.75
        assert result_msg.error_message == "Max cycles reached"
    
    def test_result_message_serialization(self):
        """Test result message JSON serialization/deserialization."""
        result_msg = AgentResultMessage(
            task_id="serialize_result",
            status="SUCCESS",
            final_state_id="final_123",
            result_data={"output": "test"},
            correlation_id="test_corr"
        )
        
        # Serialize to JSON
        json_str = result_msg.model_dump_json()
        assert "serialize_result" in json_str
        assert "SUCCESS" in json_str
        
        # Deserialize from JSON
        restored_msg = AgentResultMessage.model_validate_json(json_str)
        assert restored_msg.task_id == result_msg.task_id
        assert restored_msg.status == result_msg.status
        assert restored_msg.final_state_id == result_msg.final_state_id
        assert restored_msg.result_data == result_msg.result_data
        assert restored_msg.correlation_id == result_msg.correlation_id
    
    def test_result_message_status_types(self):
        """Test different status types."""
        statuses = ["SUCCESS", "FAILURE", "INCOMPLETE", "INVALID_MESSAGE", "TIMEOUT"]
        
        for status in statuses:
            result_msg = AgentResultMessage(
                task_id=f"task_{status.lower()}",
                status=status
            )
            assert result_msg.status == status
            assert result_msg.task_id == f"task_{status.lower()}"
    
    def test_result_message_extra_fields(self):
        """Test that extra fields are allowed."""
        data = {
            "task_id": "extra_result",
            "status": "SUCCESS",
            "custom_metric": 42,
            "execution_node": "worker_1"
        }
        
        result_msg = AgentResultMessage.model_validate(data)
        assert result_msg.task_id == "extra_result"
        assert result_msg.status == "SUCCESS"
        # Extra fields should be preserved
    
    def test_result_message_validation_errors(self):
        """Test validation errors for required fields."""
        # Missing task_id
        with pytest.raises(Exception):  # Pydantic ValidationError
            AgentResultMessage(status="SUCCESS")
        
        # Missing status
        with pytest.raises(Exception):
            AgentResultMessage(task_id="test_id")
        
        # Empty status should be valid (just test that it works)
        result_msg = AgentResultMessage(task_id="test_id", status="")
        assert result_msg.status == ""


class TestMessageIntegration:
    """Test integration between task and result messages."""
    
    def test_correlation_id_flow(self):
        """Test correlation ID flow from task to result."""
        correlation_id = "flow_test_123"
        
        # Create task message
        task_msg = AgentTaskMessage(
            task_id="integration_task",
            task_description="Test correlation flow",
            correlation_id=correlation_id
        )
        
        # Create result message with same correlation ID
        result_msg = AgentResultMessage(
            task_id=task_msg.task_id,
            status="SUCCESS",
            correlation_id=task_msg.correlation_id
        )
        
        assert task_msg.correlation_id == result_msg.correlation_id == correlation_id
        assert task_msg.task_id == result_msg.task_id
    
    def test_state_id_flow(self):
        """Test state ID flow from task to result."""
        task_msg = AgentTaskMessage(
            task_id="state_flow_task",
            task_description="Test state flow",
            initial_state_id="initial_state_456"
        )
        
        result_msg = AgentResultMessage(
            task_id=task_msg.task_id,
            status="SUCCESS",
            final_state_id="final_state_789"
        )
        
        assert result_msg.task_id == task_msg.task_id
        assert result_msg.final_state_id != task_msg.initial_state_id
        # Final state should be different from initial state
    
    def test_round_trip_serialization(self):
        """Test round-trip serialization of both message types."""
        # Task message round trip
        task_original = AgentTaskMessage(
            task_id="round_trip_task",
            task_description="Round trip test",
            initial_state_data={"nested": {"data": [1, 2, 3]}},
            agent_config={"complex": {"setting": True}},
            correlation_id="round_trip_corr"
        )
        
        task_json = task_original.model_dump_json()
        task_restored = AgentTaskMessage.model_validate_json(task_json)
        
        assert task_restored.model_dump() == task_original.model_dump()
        
        # Result message round trip
        result_original = AgentResultMessage(
            task_id="round_trip_task",
            status="SUCCESS",
            result_data={"complex": {"result": {"with": ["nested", "data"]}}},
            correlation_id="round_trip_corr"
        )
        
        result_json = result_original.model_dump_json()
        result_restored = AgentResultMessage.model_validate_json(result_json)
        
        assert result_restored.model_dump() == result_original.model_dump()