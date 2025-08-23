"""Tests for serialization formatting utilities."""

import pytest
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock
from pydantic import BaseModel
from flowlib.utils.formatting.serialization import make_serializable, format_execution_details


class SerializableModel(BaseModel):
    """Test model for serialization tests."""
    id: str
    name: str
    value: int
    timestamp: datetime
    metadata: Dict[str, Any]


class TestMakeSerializable:
    """Test serialization utilities."""
    
    def test_make_serializable_basic_types(self):
        """Test serializing basic types."""
        data = {
            "string": "test",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"}
        }
        
        result = make_serializable(data)
        assert isinstance(result, dict)
        assert result["string"] == "test"
        assert result["integer"] == 42
        assert result["float"] == 3.14
        assert result["boolean"] is True
        assert result["none"] is None
        assert result["list"] == [1, 2, 3]
        assert result["dict"]["nested"] == "value"
    
    def test_make_serializable_datetime(self):
        """Test serializing datetime objects."""
        dt = datetime(2023, 1, 1, 12, 0, 0)
        data = {
            "timestamp": dt,
            "event": "test"
        }
        
        result = make_serializable(data)
        assert isinstance(result["timestamp"], str)
        assert "2023-01-01" in result["timestamp"]
        assert result["event"] == "test"
    
    def test_make_serializable_pydantic_model(self):
        """Test serializing Pydantic models."""
        model = SerializableModel(
            id="1",
            name="test",
            value=42,
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            metadata={"key": "value"}
        )
        
        result = make_serializable(model)
        assert isinstance(result, dict)
        assert result["id"] == "1"
        assert result["name"] == "test"
        assert result["value"] == 42
        assert isinstance(result["timestamp"], str)
        assert result["metadata"]["key"] == "value"
    
    def test_make_serializable_mock_objects(self):
        """Test serializing Mock objects."""
        mock = Mock()
        mock.name = "test_mock"
        mock.value = 42
        
        data = {
            "mock": mock,
            "other": "data"
        }
        
        result = make_serializable(data)
        # Mock objects with __dict__ should be converted to dictionaries
        assert isinstance(result["mock"], dict)
        assert result["mock"]["name"] == "test_mock"
        assert result["mock"]["value"] == 42
        assert result["other"] == "data"
    
    def test_make_serializable_nested_complex(self):
        """Test serializing complex nested structures."""
        data = {
            "level1": {
                "level2": {
                    "models": [
                        SerializableModel(
                            id=str(i),
                            name=f"model_{i}",
                            value=i * 10,
                            timestamp=datetime(2023, 1, i, 12, 0, 0),
                            metadata={"index": i}
                        ) for i in range(1, 4)
                    ],
                    "timestamps": [
                        datetime(2023, 1, 1, 12, 0, 0),
                        datetime(2023, 1, 2, 12, 0, 0)
                    ]
                }
            }
        }
        
        result = make_serializable(data)
        assert isinstance(result, dict)
        assert len(result["level1"]["level2"]["models"]) == 3
        assert all(isinstance(model, dict) for model in result["level1"]["level2"]["models"])
        assert all(isinstance(ts, str) for ts in result["level1"]["level2"]["timestamps"])
    
    def test_make_serializable_edge_cases(self):
        """Test serialization edge cases."""
        # Empty structures
        assert make_serializable({}) == {}
        assert make_serializable([]) == []
        
        # None
        assert make_serializable(None) is None
        
        # Simple types
        assert make_serializable("string") == "string"
        assert make_serializable(42) == 42
        assert make_serializable(True) is True


class TestFormatExecutionDetails:
    """Test execution details formatting."""
    
    def test_format_execution_details_basic(self):
        """Test basic execution details formatting."""
        details = {
            "flow_name": "test_flow",
            "execution_time": 1.5,
            "success": True,
            "inputs": {"message": "test"},
            "outputs": {"result": "success"},
            "metadata": {"version": "1.0"}
        }
        
        formatted = format_execution_details(details)
        assert isinstance(formatted, dict)
        assert formatted["progress"] == 0  # Default when no state
        assert formatted["complete"] is False
        assert "execution_history" in formatted
    
    def test_format_execution_details_with_errors(self):
        """Test formatting execution details with errors."""
        details = {
            "flow_name": "failing_flow",
            "execution_time": 0.5,
            "success": False,
            "error": "Connection timeout",
            "inputs": {"data": "test"},
            "outputs": None,
            "metadata": {"retry_count": 3}
        }
        
        formatted = format_execution_details(details)
        assert isinstance(formatted, dict)
        assert formatted["progress"] == 0  # Default when no state
        assert formatted["complete"] is False
        assert "execution_history" in formatted
    
    def test_format_execution_details_complex(self):
        """Test formatting complex execution details."""
        details = {
            "flow_name": "complex_flow",
            "execution_time": 5.25,
            "success": True,
            "inputs": {
                "data": [1, 2, 3, 4, 5],
                "config": {
                    "batch_size": 100,
                    "timeout": 30,
                    "retry_policy": {"max_retries": 3, "backoff": "exponential"}
                }
            },
            "outputs": {
                "processed_items": 5,
                "summary": {"total": 5, "success": 5, "failed": 0},
                "timestamp": datetime(2023, 1, 1, 12, 0, 0)
            },
            "metadata": {
                "worker_id": "worker_001",
                "memory_usage": 256.5,
                "cpu_time": 2.1
            }
        }
        
        formatted = format_execution_details(details)
        assert isinstance(formatted, dict)
        assert formatted["progress"] == 0  # Default when no state
        assert formatted["complete"] is False
        assert "execution_history" in formatted
    
    def test_format_execution_details_empty(self):
        """Test formatting empty execution details."""
        formatted = format_execution_details({})
        assert isinstance(formatted, dict)
        assert formatted["error"] == "No execution details available"